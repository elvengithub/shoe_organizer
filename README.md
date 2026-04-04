# Shoe organizer

Web app for the shoe organizer. Runs on your computer; open it in a browser.

## What you need

- **Python 3** — [Download Python](https://www.python.org/downloads/). On Windows, enable **Add python.exe to PATH** when installing.

## Dependencies

These Python libraries are installed automatically from `requirements.txt` when you run `pip install -r requirements.txt`:

| Package | Role |
|--------|------|
| Flask | Web server and pages |
| opencv-python-headless | Camera / image analysis |
| numpy | Math for images |
| Pillow | Image handling |
| PyYAML | Reads `config.yaml` |
| python-dotenv | Optional environment settings |
| pyserial | ESP32 USB serial telemetry (install via `requirements.txt`) |

Extra packages for Raspberry Pi or training models are **optional** and noted as comments inside `requirements.txt`.

## How to start the project

Your download may have **two** folders named `shoe_organizer` (one inside the other). The **code** is in the **inner** folder. You can do either:

- **Option A (easiest):** Open a terminal in the **outer** folder (the one that contains a subfolder also called `shoe_organizer`). That outer folder now has its own **`run.py`** and **`requirements.txt`** that forward to the inner project — use the steps below there.
- **Option B:** `cd` into the **inner** `shoe_organizer` folder (where `src\` and `config.yaml` live) and run the same commands there.

1. Open a terminal in the folder that contains **`run.py`** and **`requirements.txt`** (outer or inner as above).
2. Run:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

On Mac or Linux, use `source .venv/bin/activate` instead of `.venv\Scripts\activate`.

3. In your browser, open **http://localhost:8080**

To stop the app: press **Ctrl+C** in the terminal.

**Testing without hardware (e.g. on a laptop):** in PowerShell, run `$env:MOCK_HARDWARE = "1"` before `python run.py`. On Mac/Linux: `export MOCK_HARDWARE=1`.
