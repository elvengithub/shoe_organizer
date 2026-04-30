# =============================================================================
# ESP32 #1 — TELEMETRY (DHT, ultrasonic, gas, POST /api/esp32/telemetry)
#
# This board drives “ESP32 live / offline” on the dashboard for slots 2–6.
#
# Two-ESP32 setup: keep EXTRA_RELAY_GPIOS = () here. On ESP32 #2 flash
# micropython/esp32_fan_controller.py as main.py (fans + 2 pumps + 3 motors).
# Step-by-step: micropython/SETUP_TWO_ESP32.md
#
# Single-ESP32 (everything on one board): set EXTRA_RELAY_GPIOS only if pins don’t
# clash with TRIG_PINS / ECHO_PINS / DHT_PINS / GAS_PINS below.
#
# THONNY: copy this file → Save as main.py on ESP32 #1 → Run.
# Wi‑Fi + SERVER_BASE below. Shell: import mip; mip.install("urequests")
# =============================================================================
from machine import Pin, ADC, time_pulse_us
import dht
import gc
import json
import sys
import time

try:
    import urequests
except ImportError:
    try:
        import requests as urequests
    except ImportError:
        urequests = None

try:
    import network
except ImportError:
    network = None

try:
    import uselect as select
except ImportError:
    import select


def _http_base(url):
    """Normalize URL (fix ttp:// typo) for urequests."""
    s = str(url or "").strip()
    if s.startswith("ttp://"):
        s = "http://" + s[6:]
    elif s.startswith("ttps://"):
        s = "https://" + s[7:]
    elif not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s.lstrip("/")
    return s.rstrip("/")


# --- Network (edit only if you change router / Pi IP) ---
WIFI_SSID = "ZTE_2.4G_2hDpYx"
WIFI_PASSWORD = "dniZYDVK"
SERVER_BASE = "http://192.168.1.16:8080"  # Flask on LAN — never use localhost here
ESP32_SECRET = ""  # optional: same string as esp32_telemetry.secret in config.yaml
SHOE_COMPARTMENT_IDS = [2, 3, 4, 5, 6]
POST_INTERVAL_MS = 900
POST_RETRIES = 3
DEBUG_TERMINAL = True
WIFI_CONNECT_TIMEOUT_MS = 30000
WIFI_RETRY_INTERVAL_MS = 15000

SERVER_BASE = _http_base(SERVER_BASE)

# Pump ug fan: kung ang Flask makakita ug sapatos (bay camera), mo-on ang duha (poll /api/esp32/actuators).
# Ilisi ang GPIO kung magbangga sa imong wiring; RELAY_ON_IS_HIGH=False kung LOW = ON sa relay module.
PUMP_GPIO = 16
FAN_GPIO = 17
ACTUATOR_POLL_MS = 500
RELAY_ON_IS_HIGH = True
# Dashboard per-slot fans: GET /api/esp32/camera-relays → extra_relay_on (length 6).
# Indices 0–4 → slots 2–6 (see shoe_organizer slot_fan_state.py); index 5 reserved — repeat a spare/off GPIO if unused.
# Leave () to disable — without this poll, Flask still stores dashboard toggles but no ESP GPIO follows them.
EXTRA_RELAYS_POLL_MS = 600
# Six storage-fan relay GPIOs, or () to skip. Must not overlap TRIG/ECHO/DHT/GAS pins above.
EXTRA_RELAY_GPIOS = ()
# Thonny Shell: type 1–6 + Enter to toggle that relay; keys 1–5 report to Flask (slots 2–6) so the dashboard matches.
SERIAL_STORAGE_FAN_KEYS = True

_extra_slot_pins = []

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

pump_out = Pin(PUMP_GPIO, Pin.OUT, value=0)
fan_out = Pin(FAN_GPIO, Pin.OUT, value=0)

temps = [None] * 5
hums = [None] * 5


def wlan_connected():
    if network is None:
        return False
    try:
        return network.WLAN(network.STA_IF).isconnected()
    except OSError:
        return False


def try_connect_wifi():
    """Return True if STA is connected. Does not raise; retries are done from main loop."""
    if network is None:
        print("WiFi: network module missing")
        return False
    if WIFI_SSID == "YOUR_SSID" or WIFI_PASSWORD == "YOUR_PASSWORD":
        return False
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return True
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    deadline = time.ticks_ms() + WIFI_CONNECT_TIMEOUT_MS
    while not wlan.isconnected() and time.ticks_ms() < deadline:
        time.sleep_ms(200)
    if wlan.isconnected():
        print("WiFi OK:", wlan.ifconfig())
        return True
    print("WiFi: connect failed — check SSID/password; will retry")
    return False


def _finite_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except (TypeError, ValueError):
        return None


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


def _relay_drive_level(on):
    v = 1 if on else 0
    if not RELAY_ON_IS_HIGH:
        v = 1 - v
    return v


def set_pump_fan(on):
    v = _relay_drive_level(bool(on))
    pump_out.value(v)
    fan_out.value(v)


def fetch_camera_relays():
    """GET /api/esp32/camera-relays — per-slot dashboard fan bits (extra_relay_on)."""
    if urequests is None or not wlan_connected():
        return None
    gc.collect()
    url = _http_base(SERVER_BASE) + "/api/esp32/camera-relays"
    try:
        r = urequests.get(url, headers={"Connection": "close"})
        try:
            code = getattr(r, "status_code", 200)
            if not (200 <= code < 300):
                return None
            txt = r.text
        finally:
            r.close()
        return json.loads(txt)
    except OSError:
        return None
    except ValueError:
        return None


def apply_extra_slot_relays(bits):
    """Drive GPIOs from Flask extra_relay_on (booleans). Safe no-op when EXTRA_RELAY_GPIOS unset."""
    if not _extra_slot_pins or not isinstance(bits, list):
        return
    for i in range(min(len(bits), len(_extra_slot_pins))):
        try:
            on = bool(bits[i])
        except (TypeError, ValueError):
            on = False
        _extra_slot_pins[i].value(_relay_drive_level(on))


def _extra_relay_logical_on(pin):
    return pin.value() == _relay_drive_level(True)


def toggle_extra_relay_index(idx):
    """Toggle relay idx 0..5; return new logical ON state."""
    p = _extra_slot_pins[idx]
    new_on = not _extra_relay_logical_on(p)
    p.value(_relay_drive_level(new_on))
    return new_on


def post_storage_fan_slot_flask(compartment_id, on):
    """POST /api/esp32/storage-fans — keeps web UI in sync with Thonny key toggles (slots 2–6)."""
    if urequests is None or not wlan_connected():
        return False
    gc.collect()
    url = _http_base(SERVER_BASE) + "/api/esp32/storage-fans"
    body = json.dumps({"compartment": int(compartment_id), "on": bool(on)})
    headers = {"Content-Type": "application/json", "Connection": "close"}
    if ESP32_SECRET:
        headers["X-ESP32-Secret"] = ESP32_SECRET
    try:
        r = urequests.post(url, data=body, headers=headers)
        try:
            code = getattr(r, "status_code", 200)
            ok = 200 <= code < 300
        finally:
            r.close()
        return ok
    except OSError:
        return False


def poll_serial_storage_fan_keys():
    """Non-blocking: Thonny keys 1–6 toggle relays; 1–5 push state to Flask."""
    if not SERIAL_STORAGE_FAN_KEYS or len(_extra_slot_pins) != 6:
        return
    try:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return
        line = sys.stdin.readline()
        if not line:
            return
        s = line.strip()
        if not s:
            return
        cmd = s[0].upper()
        if cmd not in "123456":
            return
        idx = int(cmd) - 1
        new_on = toggle_extra_relay_index(idx)
        print(">>> Relay {} -> {}".format(cmd, "ON" if new_on else "OFF"))
        if idx < 5:
            post_storage_fan_slot_flask(idx + 2, new_on)
    except Exception:
        pass


def fetch_actuators():
    """GET /api/esp32/actuators — shoe_detected / pump_on / fan_on from Flask bay camera."""
    if urequests is None or not wlan_connected():
        return None
    gc.collect()
    url = _http_base(SERVER_BASE) + "/api/esp32/actuators"
    try:
        r = urequests.get(url, headers={"Connection": "close"})
        try:
            code = getattr(r, "status_code", 200)
            if not (200 <= code < 300):
                return None
            txt = r.text
        finally:
            r.close()
        return json.loads(txt)
    except OSError:
        return None
    except ValueError:
        return None


def ping_flask():
    """GET /api/esp32/ping — confirms MicroPython can reach Flask (same base URL as telemetry)."""
    if urequests is None or not wlan_connected():
        return False
    url = _http_base(SERVER_BASE) + "/api/esp32/ping"
    try:
        r = urequests.get(url, headers={"Connection": "close"})
        try:
            code = getattr(r, "status_code", 200)
            ok = 200 <= code < 300
        finally:
            r.close()
        return ok
    except OSError:
        return False
    except ValueError:
        return False


def post_telemetry(compartments_dict):
    if urequests is None:
        print("urequests missing — Shell: import mip; mip.install('urequests')")
        return
    if not wlan_connected():
        return
    gc.collect()
    url = _http_base(SERVER_BASE) + "/api/esp32/telemetry"
    body = json.dumps({"compartments": compartments_dict})
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Connection": "close",
    }
    if ESP32_SECRET:
        headers["X-ESP32-Secret"] = ESP32_SECRET
    last_err = None
    for attempt in range(POST_RETRIES):
        try:
            r = urequests.post(url, data=body, headers=headers)
            try:
                code = getattr(r, "status_code", 200)
                data = getattr(r, "content", None)
                if data is None and hasattr(r, "text"):
                    _ = r.text
                elif data is not None:
                    _ = data
            finally:
                r.close()
            if 200 <= code < 300:
                return
            print("POST HTTP", code)
            return
        except OSError as e:
            last_err = e
            if attempt + 1 < POST_RETRIES:
                time.sleep_ms(200 * (attempt + 1))
                continue
            break
        except ValueError as e:
            print("POST URL error — SERVER_BASE must be http://...", e)
            return
    if last_err is not None:
        print("POST failed:", last_err)


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
    tc = _finite_float(t_raw)
    hc = _finite_float(h_raw)

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
    wf = "WiFi OK" if wlan_connected() else "WiFi ---"
    print(" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  {}  ".format(wf))


def main():
    if WIFI_SSID == "YOUR_SSID" or WIFI_PASSWORD == "YOUR_PASSWORD":
        print("Edit WIFI_SSID, WIFI_PASSWORD, SERVER_BASE at top")
        return
    sb = str(SERVER_BASE or "").lower()
    if "127.0.0.1" in sb or "localhost" in sb:
        print(
            "SERVER_BASE should be the Flask host LAN IP (e.g. http://192.168.1.42:8080), not localhost — ESP32 cannot reach 127.0.0.1 on your PC."
        )
    try_connect_wifi()
    if wlan_connected() and urequests and ping_flask():
        print("Flask reachable at", _http_base(SERVER_BASE))
    elif wlan_connected() and urequests:
        print("WiFi OK but Flask not reachable — check SERVER_BASE, PC firewall port 8080, and that Flask is running (host 0.0.0.0).")
    _extra_slot_pins.clear()
    if len(EXTRA_RELAY_GPIOS) == 6:
        for g in EXTRA_RELAY_GPIOS:
            try:
                _extra_slot_pins.append(Pin(int(g), Pin.OUT, value=_relay_drive_level(False)))
            except (OSError, TypeError, ValueError) as e:
                print("EXTRA_RELAY_GPIOS Pin error:", e)
                _extra_slot_pins.clear()
                break
        if len(_extra_slot_pins) == 6:
            print(
                "Per-slot relay GPIOs:",
                EXTRA_RELAY_GPIOS,
                "(dashboard → GET /api/esp32/camera-relays)",
            )
    elif len(EXTRA_RELAY_GPIOS) not in (0,):
        print("EXTRA_RELAY_GPIOS must be () or a tuple/list of exactly 6 GPIO numbers")

    last_post = 0
    last_dht_ms = 0
    last_wifi_try = time.ticks_ms()
    last_actuator_poll = 0
    last_camera_relays_poll = 0
    last_actuator_status = None
    print("\033[?25l", end="")

    while True:
        now = time.ticks_ms()
        poll_serial_storage_fan_keys()

        if not wlan_connected() and time.ticks_diff(now, last_wifi_try) >= WIFI_RETRY_INTERVAL_MS:
            last_wifi_try = now
            try_connect_wifi()
            set_pump_fan(False)

        if (
            ACTUATOR_POLL_MS > 0
            and urequests
            and wlan_connected()
            and time.ticks_diff(now, last_actuator_poll) >= ACTUATOR_POLL_MS
        ):
            last_actuator_poll = now
            act = fetch_actuators()
            if act and act.get("ok"):
                st = act.get("status")
                if st == "clean" and last_actuator_status != "clean":
                    print("clean")
                last_actuator_status = st
                on = act.get("pump_on")
                if on is None:
                    on = bool(act.get("shoe_detected"))
                set_pump_fan(bool(on))
            else:
                last_actuator_status = None
                set_pump_fan(False)

        if (
            EXTRA_RELAYS_POLL_MS > 0
            and len(_extra_slot_pins) == 6
            and urequests
            and wlan_connected()
            and time.ticks_diff(now, last_camera_relays_poll) >= EXTRA_RELAYS_POLL_MS
        ):
            last_camera_relays_poll = now
            cr = fetch_camera_relays()
            if cr and cr.get("ok") and isinstance(cr.get("extra_relay_on"), list):
                apply_extra_slot_relays(cr.get("extra_relay_on"))
            else:
                apply_extra_slot_relays([False] * 6)

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
        if (
            urequests
            and wlan_connected()
            and time.ticks_diff(now, last_post) >= POST_INTERVAL_MS
        ):
            post_telemetry(compartments)
            last_post = now

        time.sleep_ms(80 if not DEBUG_TERMINAL else 400)


if __name__ == "__main__":
    main()
