# Shoe organizer (Raspberry Pi 4)

**Stack**

- **ESP32 + MicroPython** — hardware control and **image capture** (camera module on the ESP32 or UART/SPI to a sensor). The board **POSTs JPEG frames** to the Flask server (`POST /api/camera/frame`). Flash and iterate firmware with **Thonny**.
- **Flask (Python)** — server and **decision / touch UI** on the Raspberry Pi (or dev PC). Serves pages, climate APIs, intake, and receives ESP32 images when `camera.source` is `esp32` or `prefer_esp32`.
- **OpenCV on the Flask host** — **dirt score**, **four-way shoe type** (dress / leather / sports / casual from catalog + vision), **not-shoe** rejection (**`Not a shoe.`**), and **wash** (gentle vs deep by type + soil). Tune in `config.yaml` under `vision`, `wash`, and `shoe_gate`.

Edit this repo and templates/static in **Cursor**; use **Thonny** for ESP32 MicroPython.

## Shoe vs non-shoe (phased pipeline)

All stages use the same **ROI + CLAHE** (`vision_preprocess`) as your catalogs.

1. **Silhouette gate** (`shoe_gate`) — fast reject for empty / non-blob views.
2. **Anti-face / skin trap** (`anti_face`) — OpenCV Haar (frontal + profile) and optional YCrCb skin ratio; triggers **`Not a shoe.`** (tune or disable in `config.yaml` if tan shoes are rejected).
3. **Optional TFLite binary** (`shoe_binary`) — train on PC: `pip install -r scripts/requirements-train.txt`, then `python scripts/train_shoe_binary.py` (adds shoes from `datasets/shoes`, `datasets/dirty_Shoe`, `datasets/shoe_binary/shoe` and negatives from `datasets/not_shoe`, `datasets/shoe_binary/not_shoe`). Writes `models/shoe_binary.tflite` and `models/shoe_binary_metrics.json` with a suggested threshold. Use `--align_preprocess` if training images are **full** booth frames (matches `vision_preprocess` ROI+CLAHE). Evaluate: `python scripts/eval_shoe_binary.py`. On Pi: `pip install tflite-runtime`, copy the `.tflite`, set `shoe_binary.enabled: true` and `threshold` in `config.yaml`.
4. **Not-shoe template gallery** (`not_shoe_catalog`) — add real non-shoe images under `datasets/not_shoe/` (bottle, hand, etc.); high histogram match → **`Not a shoe.`**
5. **Frame stability** (`classification_stability`) — `confirmed_is_shoe` in `/api/camera/analyze` and intake require several consistent frames (reduces flicker).
6. **Style catalog** (`shoe_catalog`) — only after the object is accepted as a shoe.
7. **Optional mining** (`debug_capture`) — set `enabled: true` to save JPEGs of rejected ROIs into `datasets/captured_rejects/`.

Six logical compartments: **1 = cleaning bay** (webcam, wash mechanism, where shoes are cleaned and classified) and **2–6 = storage**. The software layer includes the touch UI, webcam heuristics for dirt and shoe type, wash decision (hard vs soft), free-slot selection from instructor **soft switches**, **DHT-class** temperature/humidity per storage slot, **ventilation** GPIO, and **3-axis** STEP/DIR motion (NEMA 17 + GT2 for X/Y + T8 lead screw for Z). Set `limits.compartment_y_mm` to one Y position per entry in `compartments.storage_ids` (same order).

## Quick start

From a terminal, go to **this project’s root** (the folder that contains `run.py` and `requirements.txt`). If Cursor already opened `shoe-organizer-rpi`, you are already there — **do not** run `cd shoe-organizer-rpi` again (that would look for `...\shoe-organizer-rpi\shoe-organizer-rpi`, which does not exist).

```bash
# Only if you are not already inside the repo, e.g. after cloning into your home folder:
# cd path\to\shoe-organizer-rpi

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Pi
pip install -r requirements.txt
python run.py
```

Open `http://localhost:8080` on the Pi touch display (Chromium kiosk fullscreen). The UI **automatically** calls `/api/camera/analyze` to classify what is in frame (shoe vs not shoe, type, wash). With `camera.source: esp32`, run the ESP32 to upload JPEGs, or use `usb` / `prefer_esp32` with a USB webcam.

**ESP32 → Flask (example)** — after `camera.source` is `esp32` or `prefer_esp32`, post captures from MicroPython (e.g. `urequests.post("http://<PI_IP>:8080/api/camera/frame", data=jpeg_bytes, headers={"Content-Type": "image/jpeg"})`).

Optional: `MOCK_HARDWARE=1` forces GPIO mock (useful on a laptop). Without `RPi.GPIO`, pins are mocked automatically.

## Minimal hardware beyond your list

Your list is the **motion + brain + sensing core**. You still need (minimal):

- **3× stepper drivers** (e.g. A4988/DRV8825/TMC2209), **shared motor supply** (typically 12–24 V), **Pi 5 V**
- **Endstops or manual homing** — firmware assumes you home to a known origin before trusting Y targets in `config.yaml`
- **5× MOSFET or small relay module** for storage compartment fans (vents), one per slot 2–6
- **5× momentary toggles** (instructor “occupied”) with pull-ups — GPIO reads **LOW = occupied**
- **5× DHT22** (or one I2C sensor + multiplexer) for T/RH per storage slot; until wired, the app uses **mock** climate values on non-Pi or if Adafruit Blinka + `adafruit-circuitpython-dht` are not installed
- **Official Pi touchscreen or any HDMI + USB touch** — UI is a **local Flask page** sized for touch

## Pi packages (optional, for real DHT)

```bash
sudo apt install python3-pip
pip install adafruit-blinka adafruit-circuitpython-dht
```

Map `dht_board_pin` in `config.yaml` to `board.D7`-style names from your wiring.

## Tuning

- `config.yaml` — GPIO pins, `compartment_y_mm`, GT2 teeth/pitch, T8 lead and steps/rev, vision thresholds, wash thresholds.
- Vision uses **simple OpenCV heuristics** (edges + saturation). For a capstone/demo, replace `vision_service.py` with a small **TensorFlow Lite** classifier once you have labeled images.

## Safety

Steppers can **crush or snag** fingers or shoes. Add **limit switches**, **current-limited drivers**, and **E-stop** on the mechanical build; this code does not replace physical safeguards.
