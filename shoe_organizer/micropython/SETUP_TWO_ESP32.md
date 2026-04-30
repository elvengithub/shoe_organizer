# Two ESP32 boards — what runs where

## Overview

| Board | Flash this file | Role |
|--------|-----------------|------|
| **ESP32 #1** | `main.py` (from `micropython/main.py` in the repo) | DHT, ultrasonic, gas → **`POST /api/esp32/telemetry`**. This is what makes each **slot card** show **ESP32 live** or **offline** on the website. |
| **ESP32 #2** | `main.py` (copy of `micropython/esp32_fan_controller.py`) | Same behavior as your Thonny script: **A/M**, **T/P**, **1–6**, **X/Z**; **AUTO** uses **`/api/esp32/actuators`** (wash); **fans** always follow **`/api/esp32/camera-relays`**; keys **1–5** **`POST /api/esp32/storage-fans`**. |

The website **fan On/Off buttons** only work if ESP32 #2 can reach Flask and apply `extra_relay_on` from **`/api/esp32/camera-relays`**.

---

## A. Raspberry Pi / PC (Flask)

1. Install and run the shoe organizer app (`run.py` or your usual command).
2. In **`config.yaml`**:
   - **`server.host`**: `0.0.0.0` so the LAN can connect.
   - **`server.port`**: e.g. `8080` — use the **same port** in `SERVER_BASE` on both ESP32 scripts.
   - If **`esp32_telemetry.secret`** is non-empty, set the **same** string as **`ESP32_SECRET`** on **both** ESP32 #1 and ESP32 #2 scripts.
3. Windows firewall: allow inbound TCP on that port for your Pi/PC LAN IP.

---

## B. ESP32 #1 (telemetry / “online” indicator)

1. Open **`shoe_organizer/micropython/main.py`** in the repo.
2. Set **`WIFI_SSID`**, **`WIFI_PASSWORD`**, **`SERVER_BASE`** (e.g. `http://192.168.1.16:8080`), **`ESP32_SECRET`** if used.
3. For a **two-board** setup, keep **`EXTRA_RELAY_GPIOS = ()`** (empty). Fans are not on this board.
4. In Thonny: paste → **Save as `main.py` on the device** → Run.

---

## C. ESP32 #2 (pumps, motors, fans)

1. Open **`shoe_organizer/micropython/esp32_fan_controller.py`** (name is historical; it now controls **pumps + motors + fans**).
2. Set the same **`WIFI_*`**, **`SERVER_BASE`**, **`ESP32_SECRET`** as on ESP32 #1.
3. Check **GPIO** (BCM) matches your wiring:

   | Function | Default GPIOs in script |
   |----------|-------------------------|
   | Motor 1–3 | 16, 17, 18 |
   | Water pump 1 | 19 |
   | Water pump 2 | 21 |
   | Fan relays (×6) | 22, 23, 25, 26, 27, 32 |

   Edit **`MOTOR_GPIOS`**, **`PUMP1_GPIO`**, **`PUMP2_GPIO`**, **`EXTRA_RELAY_GPIOS`** if your board differs.

4. **`RELAY_ON_IS_HIGH`**: `True` if relay module turns **ON** when GPIO is **HIGH**; set `False` if your module is **active-LOW**.

5. In Thonny: paste → **Save as `main.py` on this ESP32 only** → Run.

---

## D. Frontend fan buttons

1. Open the dashboard in a browser (same LAN as the ESP32s).
2. Turn a **fan On/Off** for slot 2–6. Flask stores the state and **`/api/esp32/camera-relays`** returns **`extra_relay_on`**.
3. ESP32 #2 must poll that URL (default every **600 ms**). Then the **physical fan relays** match the buttons.

**Thonny Shell (optional):** type **`1`**–**`6`** and **Enter** to toggle fan relays **`1`**–**`6`**. Keys **`1`**–**`5`** also **`POST`** to Flask so the **website** updates for slots **2**–**`6`**.

---

## E. Wash sequence (pumps + motors)

Flask runs the timed wash sequence when the **camera** sees a **dirty shoe**. ESP32 #2 reads **`GET /api/esp32/actuators`** and uses:

- **`pump1_on`** → pump on GPIO **`PUMP1_GPIO`**
- **`pump2_on`** → pump on **`PUMP2_GPIO`**
- **`motors_on`** → all three **motor** GPIOs together  

If the request fails or Wi‑Fi drops, this script turns **pumps and motors OFF** (fans follow the fan poll rule: all off if `camera-relays` fails).

Tune timings in **`config.yaml`** under **`wash_actuators`**.

---

## F. Quick checks

| Check | What to verify |
|--------|----------------|
| Telemetry | Slot cards show **live** when ESP32 #1 posts; fix Wi‑Fi / `SERVER_BASE` / Flask if always offline. |
| Fans | Toggle on website → relays click within ~1 s; if not, ESP32 #2 `SERVER_BASE` / Wi‑Fi / `RELAY_ON_IS_HIGH` / GPIO list. |
| Wash | Dirty shoe in bay → pumps/motors follow sequence; if not, **`/api/esp32/actuators`** in a browser and GPIO mapping. |

---

## G. File names on disk

- Repo paths: **`shoe_organizer/micropython/main.py`** (ESP32 #1), **`shoe_organizer/micropython/esp32_fan_controller.py`** (ESP32 #2, save **as `main.py`** on that device).
