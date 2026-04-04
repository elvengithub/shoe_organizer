# =============================================================================
# Thonny IDE — use THIS file on the ESP32
#
# 1) Open Thonny → File → Open → browse to this main.py in the repo.
# 2) Plug in the ESP32. Bottom-right: choose "MicroPython (ESP32)".
# 3) Edit WIFI_SSID, WIFI_PASSWORD, SERVER_BASE a few lines below (your LAN IP).
# 4) In the Shell, run once if needed:  import mip; mip.install("urequests")
# 5) Run ▶ to test. When it works: File → Save as… → MicroPython device → main.py
#    (that makes it start automatically on power-up.)
# =============================================================================
"""
5 zones (HC-SR04 + DHT22 + gas) → Flask shoe_organizer at POST /api/esp32/telemetry
Body: {"compartments": {"2": {...}, "3": {...}, ...}}
"""
from machine import Pin, ADC, time_pulse_us
import dht
import json
import time

try:
    import urequests
except ImportError:
    urequests = None

try:
    import network
except ImportError:
    network = None

# ========================== CONFIGURE THESE IN THONNY ==========================
WIFI_SSID = "ZTE_2.4G_2hDpYx"
WIFI_PASSWORD = "dniZYDVK"
SERVER_BASE = "http://192.168.1.100:8080"
ESP32_SECRET = ""
SHOE_COMPARTMENT_IDS = [2, 3, 4, 5, 6]
POST_INTERVAL_MS = 900
DEBUG_TERMINAL = True
# ------------------------------------------------------------------------------

TRIG_PINS = [5, 21, 22, 25, 2]
ECHO_PINS = [18, 19, 23, 26, 15]
DHT_PINS = [4, 13, 27, 14, 12]
GAS_PINS = [34, 35, 32, 33, 39]

BASE = 450
MIN_SCAN = 2.0
MAX_SCAN = 20.0
PULSE_TIMEOUT_US = 15000

CLEAR_SCREEN = "\033[2J\033[H"
R, Y, G, W = "\033[31m", "\033[33m", "\033[32m", "\033[0m"

trigs = [Pin(p, Pin.OUT) for p in TRIG_PINS]
echos = [Pin(p, Pin.IN) for p in ECHO_PINS]
dhts = [dht.DHT22(Pin(p)) for p in DHT_PINS]
gases = [ADC(Pin(p)) for p in GAS_PINS]
for g in gases:
    g.atten(ADC.ATTN_11DB)

temps = [None] * 5
hums = [None] * 5


def get_dist(t_pin, e_pin):
    t_pin.value(0)
    time.sleep_us(2)
    t_pin.value(1)
    time.sleep_us(10)
    t_pin.value(0)
    duration = time_pulse_us(e_pin, 1, PULSE_TIMEOUT_US)
    if duration <= 0:
        return -1.0
    return (duration / 2) / 29.1


def connect_wifi():
    if network is None:
        raise RuntimeError("network module not available")
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        deadline = time.ticks_ms() + 30000
        while not wlan.isconnected() and time.ticks_ms() < deadline:
            time.sleep_ms(200)
    if not wlan.isconnected():
        raise RuntimeError("WiFi connect failed — check SSID/password")
    print("WiFi OK:", wlan.ifconfig())


def post_telemetry(compartments_dict):
    if urequests is None:
        print("urequests missing — Shell: import mip; mip.install('urequests')")
        return
    url = SERVER_BASE.rstrip("/") + "/api/esp32/telemetry"
    body = json.dumps({"compartments": compartments_dict})
    headers = {"Content-Type": "application/json"}
    if ESP32_SECRET:
        headers["X-ESP32-Secret"] = ESP32_SECRET
    try:
        r = urequests.post(url, data=body, headers=headers)
        r.close()
    except OSError as e:
        print("POST failed:", e)


def risk_and_msg(occupied, disp_lvl, cid):
    if not occupied:
        return "IDLE", "Compartment empty"
    if disp_lvl > 60:
        return "EXTREME", "Critical: toxic / odor (C{})".format(cid)
    if disp_lvl > 25:
        return "MEDIUM", "Warning: stale (C{})".format(cid)
    return "LOW", "Sanitary / clean (C{})".format(cid)


def build_zone_payload(i, dist):
    occupied = MIN_SCAN <= dist <= MAX_SCAN
    t_raw = temps[i]
    h_raw = hums[i]
    try:
        tc = float(t_raw) if t_raw is not None else None
    except (TypeError, ValueError):
        tc = None
    try:
        hc = float(h_raw) if h_raw is not None else None
    except (TypeError, ValueError):
        hc = None

    if not occupied:
        distance_cm = None
        raw_val = None
        lvl = 0
        ultrasonic_clear = True
    else:
        distance_cm = round(dist, 1)
        raw_val = gases[i].read()
        lvl = int(max(0, min(100, ((raw_val - BASE) / 2000) * 100)))
        ultrasonic_clear = False

    shoe_cid = SHOE_COMPARTMENT_IDS[i]
    risk, msg = risk_and_msg(occupied, lvl, shoe_cid)
    return {
        "temperature_c": tc,
        "humidity_pct": hc,
        "distance_cm": distance_cm,
        "ultrasonic_max_cm": MAX_SCAN,
        "ultrasonic_clear": ultrasonic_clear,
        "gas_raw": raw_val,
        "odor_level_pct": lvl,
        "occupied": occupied,
        "risk_level": risk,
        "bio_message": msg,
    }


def render_terminal(compartments_dict):
    print(CLEAR_SCREEN, end="")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   {}BIO-SECURITY (5-ZONE) -> shoe_organizer{}".format(W, W))
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for i in range(5):
        cid = SHOE_COMPARTMENT_IDS[i]
        pl = compartments_dict.get(str(cid), {})
        dist = pl.get("distance_cm")
        occ = pl.get("occupied", False)
        disp_t = pl.get("temperature_c")
        disp_h = pl.get("humidity_pct")
        disp_lvl = pl.get("odor_level_pct", 0) or 0
        risk = pl.get("risk_level", "")
        msg = pl.get("bio_message", "")
        if dist is None:
            dist_s = "--"
        else:
            dist_s = str(int(dist)) if isinstance(dist, (int, float)) else str(dist)
        if not occ:
            occ_status = "{}VACANT        {}".format(G, W)
            dt, dh = " --.-", " --.-"
        else:
            occ_status = "{}OCCUPIED ({:>2s}cm){}".format(R, dist_s, W)
            dt = "{:>5.1f}".format(float(disp_t)) if disp_t is not None else " --.-"
            dh = "{:>5.1f}".format(float(disp_h)) if disp_h is not None else " --.-"
        print("  [ SHOE SLOT {} (zone {}) ]".format(cid, i + 1))
        print("  CLIMATE  │  {}°C  │  {}% RH  ".format(dt, dh))
        print("  PRESENCE │  {:<28}".format(occ_status))
        print("  ODOR     │  [{}]  {:>3d}%".format("■" * int(disp_lvl / 5), int(disp_lvl)))
        print("  RISK     │  {}".format(risk))
        print("  MSG      │  {}".format(msg))
        if i < 4:
            print("  ──────────────────────────────────────")
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main():
    if WIFI_SSID == "YOUR_SSID" or WIFI_PASSWORD == "YOUR_PASSWORD":
        print("Edit WIFI_SSID, WIFI_PASSWORD, SERVER_BASE at top of main.py")
        return
    connect_wifi()
    last_post = 0
    last_dht_ms = 0
    print("\033[?25l", end="")

    while True:
        if time.ticks_ms() - last_dht_ms > 2000:
            for i in range(5):
                try:
                    dhts[i].measure()
                    temps[i] = dhts[i].temperature()
                    hums[i] = dhts[i].humidity()
                except OSError:
                    temps[i], hums[i] = None, None
            last_dht_ms = time.ticks_ms()

        compartments = {}
        for i in range(5):
            dist = get_dist(trigs[i], echos[i])
            cid = SHOE_COMPARTMENT_IDS[i]
            compartments[str(cid)] = build_zone_payload(i, dist)

        if DEBUG_TERMINAL:
            render_terminal(compartments)

        now = time.ticks_ms()
        if time.ticks_diff(now, last_post) >= POST_INTERVAL_MS:
            post_telemetry(compartments)
            last_post = now

        time.sleep_ms(80 if not DEBUG_TERMINAL else 400)


if __name__ == "__main__":
    main()
