# =============================================================================
# ESP32 #2 — FULL WEBSITE SYNC (MODE + WASH EVALUATION)
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
# CONFIGURATION
# -----------------------
WIFI_SSID = "ZTE_2.4G_2hDpYx"
WIFI_PASSWORD = "dniZYDVK"
SERVER_BASE = "http://192.168.1.3:8080"
WIFI_RETRY_MS = 15000
ACTUATOR_POLL_MS = 300       
CAMERA_RELAYS_POLL_MS = 300   

PUMP_19_INVERTED = False  
PUMP_21_INVERTED = False  
STARTUP_DELAY_MS = 10000 
PUMP_19_TIME = 8000      
PUMP_21_TIME = 5000      
MOTOR_EXTENSION_TIME = 20000  # 20 seconds for Deep Wash

# -----------------------
# HARDWARE SETUP
# -----------------------
RELAY_ON_IS_HIGH = False

motor_pins = [Pin(16, Pin.OUT), Pin(17, Pin.OUT), Pin(18, Pin.OUT)]
pump_pins = [Pin(19, Pin.OUT), Pin(21, Pin.OUT)]
extra_relays = [
    Pin(22, Pin.OUT), Pin(23, Pin.OUT), Pin(25, Pin.OUT),
    Pin(26, Pin.OUT), Pin(27, Pin.OUT), Pin(32, Pin.OUT),
]

def _drive(on, inverted=False):
    v = 1 if on else 0
    if not RELAY_ON_IS_HIGH: v = 1 - v
    if inverted: v = 1 - v
    return v

def set_group(pin_list, on):
    val = _drive(on)
    for p in pin_list: p.value(val)

def all_outputs_off():
    global _wash_state, _motor_active, _manual_pumps, _manual_motors
    _wash_state = "IDLE"; _motor_active = False; _manual_pumps = False; _manual_motors = False
    try:
        set_group(motor_pins, False)
        pump_pins[0].value(_drive(False, PUMP_19_INVERTED))
        pump_pins[1].value(_drive(False, PUMP_21_INVERTED))
        for r in extra_relays: r.value(_drive(False))
    except: pass

# -----------------------
# NETWORK HELPERS
# -----------------------
def wlan_connected():
    if network is None: return False
    try: return network.WLAN(network.STA_IF).isconnected()
    except: return False

def try_wifi():
    if network is None: return False
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected(): return True
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    deadline = time.ticks_ms() + 15000
    while not wlan.isconnected() and time.ticks_ms() < deadline:
        time.sleep_ms(200)
    return wlan.isconnected()

def fetch_json_get(path):
    if not wlan_connected() or urequests is None: return None
    gc.collect()
    try:
        r = urequests.get(SERVER_BASE + path, headers={"Connection": "close"}, timeout=5)
        try:
            if not (200 <= getattr(r, "status_code", 200) < 300): return None
            return json.loads(r.text)
        finally: r.close()
    except: return None

def fetch_json_post(path, data):
    if not wlan_connected() or urequests is None: return None
    gc.collect()
    try:
        r = urequests.post(SERVER_BASE + path, 
                           headers={"Connection": "close", "Content-Type": "application/json"},
                           data=json.dumps(data), 
                           timeout=5)
        try:
            if not (200 <= getattr(r, "status_code", 200) < 300): return None
            return json.loads(r.text)
        finally: r.close()
    except: return None

def post_to_server(path, data):
    if not wlan_connected() or urequests is None: return
    try:
        r = urequests.post(SERVER_BASE + path, data=json.dumps(data), headers={"Connection": "close"}, timeout=5)
        r.close()
    except: pass

# -----------------------
# LOGIC
# -----------------------
def start_wash_sequence(wash_mode="soft"):
    global _wash_state, _wash_timer, _p19_count, _p21_count, _motor_active, _last_cd
    global _p19_max, _p21_max, _has_extension
    
    if wash_mode == "none":
        print(">>> Evaluation: NO WASH needed.")
        return

    print("\n>>> [HW] {} TRIGGERED (Interrupting if active)".format(wash_mode.upper()))
    _wash_state = "DELAY"
    _wash_timer = time.ticks_ms()
    _p19_count = 0
    _p21_count = 0
    _motor_active = False
    _last_cd = -1
    
    # Configure cycles based on mode
    if wash_mode == "hard":
        _p19_max = 4
        _p21_max = 3
        _has_extension = True
    else: # Gentle Wash
        _p19_max = 3
        _p21_max = 2
        _has_extension = False

def apply_wash_outputs(data, ignore_master=False):
    if not data or not data.get("ok"): return
    seq_running = (str(data.get("wash_sequence_state", "")).lower() == "running")
    shoe = bool(data.get("shoe_detected", False))
    mode = str(data.get("wash_mode", "soft")).lower()
    global _last_seq_running, _last_trigger_id
    trigger_id = int(data.get("wash_trigger_id", 0))
    
    # Check if this is a brand new manual click
    if trigger_id != _last_trigger_id:
        if seq_running:
            _last_trigger_id = trigger_id
            print(">>> [SYNC] Server says NEW TRIGGER: {} (id={})".format(mode, trigger_id))
            start_wash_sequence(mode)
        else:
            # Server says this trigger is already done. Sync our ID so we don't loop.
            _last_trigger_id = trigger_id
            print(">>> [SYNC] Skipping old trigger (id={})".format(trigger_id))
            
    # Or check if auto started it
    # Or check if auto started it
    elif (shoe and not ignore_master) or (seq_running and not _last_seq_running):
        if _wash_state == "IDLE":
            print(">>> [SYNC] Server says {} (running={})".format(mode, seq_running))
            start_wash_sequence(mode)
            # If server is already deep into its countdown, we should jump past our local delay
            # but only if the server is actually running the sequence.
            # For now, start_wash_sequence is fine as it initializes the local timer.
            pass

    # (Sync stop removed to allow local sequence completion without interruption)
            
    _last_seq_running = seq_running

# --- STATE ---
MODE = "MANUAL"
_last_wifi_try = 0; _last_act_poll = 0; _last_fan_poll = 0
_wash_state = "IDLE"; _wash_timer = 0; _p19_count = 0; _p21_count = 0; _motor_active = False
_manual_motors = False; _manual_pumps = False; _last_cd = -1
_p19_max = 3; _p21_max = 2; _has_extension = False
_last_seq_running = False; _last_trigger_id = 0; _last_data = {}

all_outputs_off()

def main():
    global MODE, _last_wifi_try, _last_act_poll, _last_fan_poll
    global _wash_state, _wash_timer, _p19_count, _p21_count, _motor_active, _last_cd
    global _manual_motors, _manual_pumps, _p19_max, _p21_max, _has_extension, _last_data
    
    print("\n" + "="*50)
    print("   ESP32 MASTER CONTROLLER — DYNAMIC WASH SYNC")
    print("="*50)

    try:
        while True:
            now = time.ticks_ms()

            if not wlan_connected() and time.ticks_diff(now, _last_wifi_try) >= WIFI_RETRY_MS:
                _last_wifi_try = now; try_wifi()

            # --- THONNY INPUT ---
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().upper()
                if line:
                    cmd = line[0]
                    if cmd == "A": 
                        MODE = "AUTO"; post_to_server("/api/esp32/mode", {"mode": "AUTO"})
                    elif cmd == "M": 
                        MODE = "MANUAL"; post_to_server("/api/esp32/mode", {"mode": "MANUAL"})
                    elif cmd == "T":
                        _manual_motors = not _manual_motors
                        post_to_server("/api/motors", {"on": _manual_motors})
                    elif cmd == "P":
                        _manual_pumps = not _manual_pumps
                        post_to_server("/api/pumps", {"on": _manual_pumps})
                    elif cmd == "W": start_wash_sequence("soft") # Default Thonny manual wash
                    elif cmd == "D": start_wash_sequence("hard") # Manual Deep Wash
                    elif cmd in ("X", "Z"):
                        all_outputs_off()
                        post_to_server("/api/stop-all", {})

            # --- NETWORK SYNC ---
            if time.ticks_diff(now, _last_act_poll) >= ACTUATOR_POLL_MS:
                _last_act_poll = now
                # Sync our local state back to server so it follows OUR timer
                sync_payload = {
                    "wash_state": _wash_state,
                    "countdown": 0
                }
                if _wash_state == "DELAY":
                    sync_payload["countdown"] = (STARTUP_DELAY_MS - (time.ticks_diff(now, _wash_timer))) // 1000
                
                data = fetch_json_post("/api/esp32/actuators", sync_payload)
                if data:
                    _last_data = data
                    srv_mode = data.get("mode")
                    if srv_mode and srv_mode != MODE:
                        print(">>> SYNC: MODE -> {}".format(srv_mode))
                        MODE = srv_mode
                    
                    _manual_motors = bool(data.get("manual_motors_on"))
                    _manual_pumps = bool(data.get("manual_pumps_on"))
                    
                    apply_wash_outputs(data, ignore_master=(MODE != "AUTO"))
                else:
                    # Optional: print periodic warning if server is unreachable
                    if now % 5000 < 100: print(">>> [NET] Server unreachable (actuators)")

            if time.ticks_diff(now, _last_fan_poll) >= CAMERA_RELAYS_POLL_MS:
                _last_fan_poll = now
                cr = fetch_json_get("/api/esp32/camera-relays")
                if cr:
                    bits = cr.get("extra_relay_on")
                    if isinstance(bits, list):
                        for i in range(min(len(bits), len(extra_relays))):
                            extra_relays[i].value(_drive(bool(bits[i])))

            # --- ENGINE ---
            if _wash_state != "IDLE":
                elapsed = time.ticks_diff(now, _wash_timer)
                if _wash_state == "DELAY":
                    # Sync countdown with server if available
                    srv_cd = _last_data.get("wash_sequence_countdown")
                    if srv_cd is not None and srv_cd > 0:
                        rem = int(srv_cd)
                    else:
                        rem = (STARTUP_DELAY_MS - elapsed) // 1000
                        
                    if rem != _last_cd:
                        if rem >= 0: print(">>> Starting in {}s...".format(rem))
                        _last_cd = rem
                    
                    # Force jump if server is already running
                    if srv_cd == 0 and _last_data.get('wash_sequence_state') == 'running' and _last_data.get('pump1_on'):
                         elapsed = STARTUP_DELAY_MS # Force completion of delay
                         
                    if elapsed >= STARTUP_DELAY_MS:
                        _wash_state = "P19"; _wash_timer = now; _p19_count += 1
                        _motor_active = True
                        print(">>> [WASH] MOTOR + PUMP 19 (Cycle {}/{})".format(_p19_count, _p19_max))
                
                elif _wash_state == "P19":
                    if elapsed >= PUMP_19_TIME:
                        if _p21_count < _p21_max:
                            _wash_state = "P21"; _wash_timer = now; _p21_count += 1
                            print(">>> [WASH] MOTOR + PUMP 21 (Cycle {}/{})".format(_p21_count, _p21_max))
                        elif _p19_count < _p19_max:
                            _wash_state = "P19"; _wash_timer = now; _p19_count += 1
                            print(">>> [WASH] MOTOR + PUMP 19 (Cycle {}/{})".format(_p19_count, _p19_max))
                        else:
                            if _has_extension:
                                _wash_state = "FINAL_MOTORS"; _wash_timer = now
                                print(">>> [WASH] MOTOR EXTENSION (20s)...")
                            else:
                                all_outputs_off(); print(">>> WASH COMPLETE.")
                
                elif _wash_state == "P21":
                    if elapsed >= PUMP_21_TIME:
                        if _p19_count < _p19_max:
                            _wash_state = "P19"; _wash_timer = now; _p19_count += 1
                            print(">>> [WASH] MOTOR + PUMP 19 (Cycle {}/{})".format(_p19_count, _p19_max))
                        else:
                            if _has_extension:
                                _wash_state = "FINAL_MOTORS"; _wash_timer = now
                                print(">>> [WASH] MOTOR EXTENSION (20s)...")
                            else:
                                all_outputs_off(); print(">>> WASH COMPLETE.")

                elif _wash_state == "FINAL_MOTORS":
                    if elapsed >= MOTOR_EXTENSION_TIME:
                        all_outputs_off(); print(">>> DEEP WASH COMPLETE.")

                # Pin Drive - combine local and server states for robust timing
                if _wash_state == "DELAY":
                    # Force OFF during countdown for safety/sync
                    pump_pins[0].value(_drive(False, PUMP_19_INVERTED))
                    pump_pins[1].value(_drive(False, PUMP_21_INVERTED))
                    set_group(motor_pins, False)
                else:
                    # LOCAL states
                    loc_p1 = (_wash_state == "P19")
                    loc_p2 = (_wash_state == "P21")
                    loc_m  = _motor_active or (_wash_state == "FINAL_MOTORS")
                    
                    # SERVER states
                    srv_p1 = _last_data.get('pump1_on', False)
                    srv_p2 = _last_data.get('pump2_on', False)
                    srv_m  = _last_data.get('motors_on', False)
                    
                    # COMBINE (Stay ON if either says ON)
                    pump_pins[0].value(_drive(loc_p1 or srv_p1, PUMP_19_INVERTED))
                    pump_pins[1].value(_drive(loc_p2 or srv_p2, PUMP_21_INVERTED))
                    set_group(motor_pins, loc_m or srv_m)
            else:
                # Apply Manual overrides
                set_group(motor_pins, _manual_motors)
                pump_pins[0].value(_drive(_manual_pumps, PUMP_19_INVERTED))
                pump_pins[1].value(_drive(_manual_pumps, PUMP_21_INVERTED))

            time.sleep_ms(20)
    except KeyboardInterrupt:
        print("\n>>> Stopped.")
    finally:
        all_outputs_off()

if __name__ == "__main__":
    main()
