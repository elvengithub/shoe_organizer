# MicroPython (Thonny): pump + fan with MANUAL keys or AUTO from Flask bay camera.
# Same Wi‑Fi as the PC running Flask. Change SERVER_BASE (and WiFi) at top.
# Save as main.py on the ESP32 if this is your only script.
from machine import Pin
import sys
import time
import gc
import json

try:
    import uselect as select
except ImportError:
    import select

try:
    import urequests
except ImportError:
    urequests = None

try:
    import network
except ImportError:
    network = None

# -----------------------
# NETWORK + FLASK (AUTO)
# -----------------------
WIFI_SSID = "YOUR_SSID"
WIFI_PASSWORD = "YOUR_PASSWORD"
# LAN URL of the machine running shoe_organizer Flask (not localhost).
SERVER_BASE = "http://192.168.1.14:8080"
WIFI_CONNECT_TIMEOUT_MS = 30000
WIFI_RETRY_MS = 15000

# -----------------------
# PINS
# -----------------------
PUMP = Pin(16, Pin.OUT)
FAN = Pin(17, Pin.OUT)
# Set False if your relay module turns ON when GPIO is LOW.
RELAY_ON_IS_HIGH = True

PUMP.value(0)
FAN.value(0)


def _http_base(url):
    s = str(url or "").strip()
    if s.startswith("ttp://"):
        s = "http://" + s[6:]
    elif s.startswith("ttps://"):
        s = "https://" + s[7:]
    elif not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s.lstrip("/")
    return s.rstrip("/")


def _drive(on):
    v = 1 if on else 0
    if not RELAY_ON_IS_HIGH:
        v = 1 - v
    return v


def set_pump(on):
    PUMP.value(_drive(on))


def set_fan(on):
    FAN.value(_drive(on))


def wlan_connected():
    if network is None:
        return False
    try:
        return network.WLAN(network.STA_IF).isconnected()
    except OSError:
        return False


def try_wifi():
    if network is None or WIFI_SSID == "YOUR_SSID":
        return False
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return True
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    deadline = time.ticks_ms() + WIFI_CONNECT_TIMEOUT_MS
    while not wlan.isconnected() and time.ticks_ms() < deadline:
        time.sleep_ms(200)
    return wlan.isconnected()


# -----------------------
# MODES
# -----------------------
MODE = "MANUAL"

SAFETY_LIMIT = 10

pump_on_time = None
fan_on_time = None

_last_wifi_try = 0
_last_actuator_status_printed = None


def fetch_actuators_json():
    """GET /api/esp32/actuators — full JSON (pump_on false when shoe is clean / no wash)."""
    if urequests is None or not wlan_connected():
        return None
    gc.collect()
    url = _http_base(SERVER_BASE) + "/api/esp32/actuators"
    try:
        r = urequests.get(url, headers={"Connection": "close"})
        try:
            if getattr(r, "status_code", 200) < 200 or getattr(r, "status_code", 200) >= 300:
                return None
            return json.loads(r.text)
        finally:
            r.close()
    except (OSError, ValueError, TypeError):
        return None


print("SYSTEM READY")
print("A = AUTO (Flask shoe -> pump+fan) | M = MANUAL")
print("P = Toggle Pump | F = Toggle Fan")
print("X = ALL ON | Z = ALL OFF")

if urequests is None:
    print("Install urequests: import mip; mip.install('urequests')")


def auto_logic():
    """AUTO: pump+fan on only when Flask says pump_on (off for clean shoe / no wash)."""
    global _last_wifi_try, _last_actuator_status_printed
    now = time.ticks_ms()
    if not wlan_connected():
        if _last_wifi_try == 0 or time.ticks_diff(now, _last_wifi_try) >= WIFI_RETRY_MS:
            _last_wifi_try = now
            try_wifi()
        _last_actuator_status_printed = None
        return False
    data = fetch_actuators_json()
    if not data or not data.get("ok"):
        _last_actuator_status_printed = None
        return False
    st = data.get("status")
    if st == "clean" and _last_actuator_status_printed != "clean":
        print("clean")
    _last_actuator_status_printed = st
    on = data.get("pump_on")
    if on is None:
        on = bool(data.get("shoe_detected"))
    return bool(on)


def safety_check():
    global pump_on_time, fan_on_time

    now = time.time()

    if PUMP.value() == _drive(True) and pump_on_time is not None:
        if now - pump_on_time > SAFETY_LIMIT:
            set_pump(False)
            pump_on_time = None
            print("PUMP AUTO OFF (safety)")

    if FAN.value() == _drive(True) and fan_on_time is not None:
        if now - fan_on_time > SAFETY_LIMIT:
            set_fan(False)
            fan_on_time = None
            print("FAN AUTO OFF (safety)")


def pump_is_on():
    return PUMP.value() == _drive(True)


def fan_is_on():
    return FAN.value() == _drive(True)


if WIFI_SSID != "YOUR_SSID" and WIFI_PASSWORD != "YOUR_PASSWORD":
    if try_wifi():
        print("WiFi OK — Flask:", _http_base(SERVER_BASE))
    else:
        print("WiFi: connect failed — will retry in AUTO")

while True:
    if select.select([sys.stdin], [], [], 0)[0]:
        cmd = sys.stdin.readline().strip().upper()

        if cmd == "A":
            MODE = "AUTO"
            print("MODE -> AUTO (Flask /api/esp32/actuators)")

        elif cmd == "M":
            MODE = "MANUAL"
            print("MODE -> MANUAL")

        elif cmd == "P":
            if not pump_is_on():
                set_pump(True)
                pump_on_time = time.time()
                print("PUMP ON")
            else:
                set_pump(False)
                pump_on_time = None
                print("PUMP OFF")

        elif cmd == "F":
            if not fan_is_on():
                set_fan(True)
                fan_on_time = time.time()
                print("FAN ON")
            else:
                set_fan(False)
                fan_on_time = None
                print("FAN OFF")

        elif cmd == "X":
            set_pump(True)
            set_fan(True)
            pump_on_time = time.time()
            fan_on_time = time.time()
            print("ALL ON")

        elif cmd == "Z":
            set_pump(False)
            set_fan(False)
            pump_on_time = None
            fan_on_time = None
            print("ALL OFF")

    if MODE == "AUTO":
        if auto_logic():
            set_pump(True)
            set_fan(True)
            if pump_on_time is None:
                pump_on_time = time.time()
            if fan_on_time is None:
                fan_on_time = time.time()
        else:
            set_pump(False)
            set_fan(False)
            pump_on_time = None
            fan_on_time = None

    safety_check()
    time.sleep(0.5)
