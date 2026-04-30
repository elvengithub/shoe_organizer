# =============================================================================
# ESP32 #2 — MANUAL/AUTO + Flask (fans, wash sequence)
#
# MANUAL: motors & pumps are local (T/P/X). Fans always follow GET /api/esp32/camera-relays
#         when Wi‑Fi + Flask respond (website On/Off works). On failed GET in MANUAL, fan
#         GPIOs are left unchanged (avoids wiping Thonny toggles when Wi‑Fi drops).
# AUTO:   wash from /api/esp32/actuators; fans still from camera-relays.
#
# Stop (Thonny) / Ctrl+C: all outputs forced OFF.
# =============================================================================
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
# Wi‑Fi & Flask
# -----------------------
WIFI_SSID = "ZTE_2.4G_2hDpYx"
WIFI_PASSWORD = "dniZYDVK"
SERVER_BASE = "http://192.168.1.27:8080"
ESP32_SECRET = ""
WIFI_RETRY_MS = 15000
ACTUATOR_POLL_MS = 500
CAMERA_RELAYS_POLL_MS = 600

# -----------------------
# WASH TIMINGS
# -----------------------
MOTOR_ON_TIME = 5000   
MOTOR_OFF_TIME = 10000 
PUMP_19_TIME = 8000
PUMP_21_TIME = 5000
PUMP_19_MAX_CYCLES = 3
PUMP_21_MAX_CYCLES = 2

# -----------------------
# PINS
# -----------------------
RELAY_ON_IS_HIGH = False

motor_pins = [Pin(16, Pin.OUT), Pin(17, Pin.OUT), Pin(18, Pin.OUT)]
pump_pins = [Pin(19, Pin.OUT), Pin(21, Pin.OUT)]
extra_relays = [
    Pin(22, Pin.OUT),
    Pin(23, Pin.OUT),
    Pin(25, Pin.OUT),
    Pin(26, Pin.OUT),
    Pin(27, Pin.OUT),
    Pin(32, Pin.OUT),
]


def _http_base(url):
    s = str(url or "").strip()
    if s.startswith("ttp://"):
        s = "http://" + s[6:]
    elif not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s.lstrip("/")
    return s.rstrip("/")


SERVER_BASE = _http_base(SERVER_BASE)


def _drive(on):
    v = 1 if on else 0
    if not RELAY_ON_IS_HIGH:
        v = 1 - v
    return v


def set_group(pin_list, on):
    val = _drive(on)
    for p in pin_list:
        p.value(val)


def set_pump_pair(p1_on, p2_on):
    pump_pins[0].value(_drive(bool(p1_on)))
    pump_pins[1].value(_drive(bool(p2_on)))


def all_outputs_off():
    """Logical OFF for every relay/motor/pump (safe state when script stops)."""
    try:
        set_group(motor_pins, False)
        set_pump_pair(False, False)
        for r in extra_relays:
            r.value(_drive(False))
    except (OSError, RuntimeError):
        pass


def toggle_pin(p, name):
    new_state = not (p.value() == _drive(True))
    p.value(_drive(new_state))
    print(">>> Relay {} is now {}".format(name, "ON" if new_state else "OFF"))
    return new_state


def group_is_on(pin_list):
    return pin_list[0].value() == _drive(True)


def wlan_connected():
    if network is None:
        return False
    try:
        return network.WLAN(network.STA_IF).isconnected()
    except OSError:
        return False


def try_wifi():
    if network is None:
        return False
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return True
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    deadline = time.ticks_ms() + 20000
    while not wlan.isconnected() and time.ticks_ms() < deadline:
        time.sleep_ms(200)
    return wlan.isconnected()


def fetch_json_get(path):
    if not wlan_connected() or urequests is None:
        return None
    gc.collect()
    url = SERVER_BASE + path
    try:
        r = urequests.get(url, headers={"Connection": "close"})
        try:
            if not (200 <= getattr(r, "status_code", 200) < 300):
                return None
            txt = r.text
            return json.loads(txt)
        finally:
            r.close()
    except (OSError, ValueError, TypeError):
        return None


def fetch_actuators_json():
    return fetch_json_get("/api/esp32/actuators")


def fetch_camera_relays_json():
    return fetch_json_get("/api/esp32/camera-relays")


def apply_wash_outputs(data, is_manual=False):
    global _motor_logical_on, _pump_sequence_running, _pump_timer_start, _pump_step, _pump_cycle_count
    if not data or not data.get("ok"):
        # In manual mode, don't force OFF on network failure to avoid flickering
        if not is_manual:
            _motor_logical_on = False
            _pump_sequence_running = False
        return
    
    # Look for specific triggers or master triggers
    master = bool(data.get("shoe_detected") or data.get("all_relays_on"))
    mot_on = bool(data.get("motors_on") or master)
    pumps_on = bool(data.get("pumps_on") or data.get("pump_on") or master)
    
    # Individual actuator overrides
    if not mot_on:
        mot_on = bool(data.get("motors_on", False))
    if not pumps_on:
        pumps_on = bool(data.get("pump1_on", False) or data.get("pump2_on", False))

    # Motor State
    _motor_logical_on = mot_on

    # Pump Sequence State
    if pumps_on:
        if not _pump_sequence_running:
            _pump_sequence_running = True
            _pump_timer_start = time.ticks_ms()
            _pump_step = 0
            _pump_cycle_count = 0
    else:
        _pump_sequence_running = False


def apply_extra_relays_from_server(data):
    if not data or not data.get("ok"):
        for r in extra_relays:
            r.value(_drive(False))
        return
    bits = data.get("extra_relay_on")
    if not isinstance(bits, list):
        return
    for i in range(min(len(bits), len(extra_relays))):
        try:
            on = bool(bits[i])
        except (TypeError, ValueError):
            on = False
        extra_relays[i].value(_drive(on))


def post_storage_fan_slot(compartment_id, on):
    if not wlan_connected() or urequests is None:
        return False
    gc.collect()
    url = SERVER_BASE + "/api/esp32/storage-fans"
    body = json.dumps({"compartment": int(compartment_id), "on": bool(on)})
    headers = {"Content-Type": "application/json", "Connection": "close"}
    if ESP32_SECRET:
        headers["X-ESP32-Secret"] = ESP32_SECRET
    try:
        r = urequests.post(url, data=body, headers=headers)
        try:
            ok = 200 <= getattr(r, "status_code", 200) < 300
        finally:
            r.close()
        return ok
    except OSError:
        return False


MODE = "MANUAL"
_last_wifi_try = 0
_last_act_poll = 0
_last_fan_poll = 0

# Sequence State
_pump_sequence_running = False
_pump_timer_start = 0
_pump_step = 0
_pump_cycle_count = 0

_motor_timer_start = 0
_motor_is_in_high_phase = True
_motor_logical_on = False

all_outputs_off()

print("-" * 40)
print("SYSTEM READY — Flask + local control")
print("MODES: A=AUTO  M=MANUAL")
print("MANUAL: T/P/X = motors+pumps local; fans follow website (camera-relays).")
print("AUTO:   wash + fans from Flask.")
print("Stop script: all outputs -> OFF")
print("-" * 40)


def main():
    global MODE, _last_wifi_try, _last_act_poll, _last_fan_poll
    global _pump_sequence_running, _pump_timer_start, _pump_step, _pump_cycle_count
    global _motor_timer_start, _motor_is_in_high_phase, _motor_logical_on
    try:
        while True:
            now = time.ticks_ms()

            if not wlan_connected() and time.ticks_diff(now, _last_wifi_try) >= WIFI_RETRY_MS:
                _last_wifi_try = now
                try_wifi()

            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().upper()
                if line:
                    cmd = line[0]
                    if cmd == "A":
                        MODE = "AUTO"
                        print(">>> MODE: AUTO")
                    elif cmd == "M":
                        MODE = "MANUAL"
                        print(">>> MODE: MANUAL")
                    elif cmd == "T":
                        _motor_logical_on = not _motor_logical_on
                        _motor_timer_start = now
                        _motor_is_in_high_phase = True
                        print(">>> Motors Alternating Cycle:", "STARTED" if _motor_logical_on else "STOPPED")
                    elif cmd == "P":
                        _pump_sequence_running = not _pump_sequence_running
                        if _pump_sequence_running:
                            _pump_timer_start = now
                            _pump_step = 0
                            _pump_cycle_count = 0
                            print(">>> Pumps: STARTING SEQUENCE")
                        else:
                            set_group(pump_pins, False)
                            print(">>> Pumps: STOPPED")
                    elif cmd in ("X", "Z"):
                        _motor_logical_on = False
                        _pump_sequence_running = False
                        all_outputs_off()
                        print(">>> ALL SYSTEMS STOPPED")
                    elif cmd in "123456":
                        idx = int(cmd) - 1
                        new_on = toggle_pin(extra_relays[idx], cmd)
                        if idx < 5:
                            post_storage_fan_slot(idx + 2, new_on)

            if MODE == "AUTO":
                if time.ticks_diff(now, _last_act_poll) >= ACTUATOR_POLL_MS:
                    _last_act_poll = now
                    if wlan_connected():
                        apply_wash_outputs(fetch_actuators_json())
                    else:
                        apply_wash_outputs(None)

            # Fans: always poll dashboard state (same API the website updates).
            if time.ticks_diff(now, _last_fan_poll) >= CAMERA_RELAYS_POLL_MS:
                _last_fan_poll = now
                if wlan_connected():
                    cr = fetch_camera_relays_json()
                    if cr is not None:
                        apply_extra_relays_from_server(cr)
                        # Also check for motor/pump manual overrides in MANUAL mode
                        if MODE == "MANUAL":
                            apply_wash_outputs(cr, is_manual=True)
                    elif MODE == "AUTO":
                        apply_extra_relays_from_server(None)
                else:
                    if MODE == "AUTO":
                        apply_extra_relays_from_server(None)

            # --- THE ALTERNATING ENGINE ---
            
            # 1. PUMP SEQUENCE
            if _pump_sequence_running:
                if _pump_cycle_count < PUMP_19_MAX_CYCLES:
                    elapsed_pump = time.ticks_diff(now, _pump_timer_start)
                    current_threshold = PUMP_19_TIME if _pump_step == 0 else PUMP_21_TIME
                    
                    if elapsed_pump >= current_threshold:
                        _pump_timer_start = now
                        if _pump_step == 0:
                            _pump_step = 1  # Move to Pin 21
                        else:
                            _pump_step = 0  # Move back to Pin 19
                            _pump_cycle_count += 1
                    
                    # PIN 19 Logic
                    p19_on = (_pump_step == 0 and _pump_cycle_count < PUMP_19_MAX_CYCLES)
                    # PIN 21 Logic
                    p21_on = (_pump_step == 1 and _pump_cycle_count < PUMP_21_MAX_CYCLES)

                    pump_pins[0].value(_drive(p19_on))
                    pump_pins[1].value(_drive(p21_on))
                else:
                    set_group(pump_pins, False)
                    _pump_sequence_running = False
                    print(">>> Pump Sequence Finished.")
            else:
                if not group_is_on(pump_pins):
                    set_group(pump_pins, False)

            # 2. MOTOR CYCLE
            if _motor_logical_on:
                elapsed_motor = time.ticks_diff(now, _motor_timer_start)
                if _motor_is_in_high_phase:
                    if not group_is_on(motor_pins): set_group(motor_pins, True)
                    if elapsed_motor >= MOTOR_ON_TIME:
                        _motor_is_in_high_phase = False
                        _motor_timer_start = now
                        set_group(motor_pins, False)
                else:
                    if elapsed_motor >= MOTOR_OFF_TIME:
                        _motor_is_in_high_phase = True
                        _motor_timer_start = now
                        set_group(motor_pins, True)
            else:
                if group_is_on(motor_pins):
                    set_group(motor_pins, False)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n>>> Stopped by user — turning all outputs OFF")
    finally:
        all_outputs_off()
        print(">>> All relays/motors/pumps OFF (safe state)")


if __name__ == "__main__":
    main()

