"""
ESP32 (30-pin DevKit) -> Flask shoe-organizer via USB serial.
No Wi-Fi. Prints one JSON line per loop (serial_bridge on PC reads COM port).

Occupancy = object between NEAR_CM and MAX_RANGE cm (tune MAX_RANGE to your compartment depth).
Farther echoes (walls) are ignored for display so the UI does not show misleading 150+ cm values.

Save as main.py on the ESP32 via Thonny.
"""
from machine import Pin, ADC, time_pulse_us
import dht
import time
import json

# --- Hardware ---
gas_sensor = ADC(Pin(34))
gas_sensor.atten(ADC.ATTN_11DB)
dht_sensor = dht.DHT22(Pin(4))
trig = Pin(5, Pin.OUT)
echo = Pin(18, Pin.IN)

# --- Ultrasonic (HC-SR04 style): tune to compartment 2 geometry ---
NEAR_CM = 2.0          # ignore closer (ringing / blind zone)
MAX_RANGE = 75.0       # max distance that still counts as "in bay" for occupancy + display
PULSE_TIMEOUT_US = 24000  # ~4 m round-trip cap; avoids bogus short timeouts
MEDIAN_SAMPLES = 5     # odd count; reduces spike noise
LOOP_MS = 800

# Gas ADC baseline (calibrate in fresh air for your MQ-style sensor)
BASE = 450

# DHT smoothing (0..1; higher = react faster, lower = steadier display)
DHT_EMA_ALPHA = 0.35

last_dht_read = 0
cur_t, cur_h = 0.0, 0.0
sm_t, sm_h = None, None


def get_distance_once():
    trig.value(0)
    time.sleep_us(2)
    trig.value(1)
    time.sleep_us(10)
    trig.value(0)
    duration = time_pulse_us(echo, 1, PULSE_TIMEOUT_US)
    if duration <= 0:
        return -1.0
    return (duration / 2.0) / 29.1


def median_distance():
    buf = []
    for _ in range(MEDIAN_SAMPLES):
        buf.append(get_distance_once())
        time.sleep_ms(12)
    buf.sort()
    return buf[len(buf) // 2]


def risk_from_logic(lvl, rh, occupied):
    if not occupied:
        return "IDLE", "Compartment empty"
    if lvl > 60 and rh > 70:
        return "EXTREME", "Critical: bacterial risk (odor + humidity)"
    if lvl > 55:
        return "HIGH", "Toxic: concentrated odor"
    if rh > 80:
        return "MEDIUM", "Warning: moisture trap"
    if lvl > 25:
        return "MODERATE", "Stale / unclean"
    return "LOW", "Sanitary / clean"


while True:
    if time.ticks_ms() - last_dht_read > 2000:
        try:
            dht_sensor.measure()
            t_new = dht_sensor.temperature()
            h_new = dht_sensor.humidity()
            cur_t, cur_h = t_new, h_new
            if sm_t is None:
                sm_t, sm_h = t_new, h_new
            else:
                a = DHT_EMA_ALPHA
                sm_t = a * t_new + (1 - a) * sm_t
                sm_h = a * h_new + (1 - a) * sm_h
            last_dht_read = time.ticks_ms()
        except OSError:
            pass

    dist = median_distance()
    occupied = NEAR_CM < dist <= MAX_RANGE

    # Only report cm when echo is inside the bay window — avoids confusing "196 cm" wall readings
    if NEAR_CM < dist <= MAX_RANGE:
        distance_cm = round(dist, 1)
    else:
        distance_cm = None

    if not occupied:
        raw_val = None
        lvl = 0
    else:
        raw_val = gas_sensor.read()
        lvl = int(max(0, min(100, ((raw_val - BASE) / 2000) * 100)))

    risk, msg = risk_from_logic(lvl, sm_h if sm_h is not None else cur_h, occupied)

    disp_t = round(sm_t if sm_t is not None else cur_t, 1)
    disp_h = round(sm_h if sm_h is not None else cur_h, 1)

    payload = {
        "temperature_c": disp_t,
        "humidity_pct": disp_h,
        "distance_cm": distance_cm,
        "ultrasonic_max_cm": MAX_RANGE,
        "ultrasonic_clear": not occupied,
        "gas_raw": raw_val,
        "odor_level_pct": lvl,
        "occupied": occupied,
        "risk_level": risk,
        "bio_message": msg,
    }
    print(json.dumps(payload))
    time.sleep_ms(LOOP_MS)
