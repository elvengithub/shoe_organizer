# Shoe organizer

Flask app for the shoe cleaning bay: camera view, shoe checks, wash hints, and slot status. Use a PC for development or a Raspberry Pi with the hardware.

The Python app and `requirements.txt` live in the **`shoe_organizer`** subfolder next to this file (that folder contains `src/`, `config.yaml`, and `run.py`).

## What to install

You need Python 3.10 or newer (3.11+ is fine). On Windows, install Python from python.org and turn on “Add python.exe to PATH”.

Open a terminal in this repository root, create a virtual environment, and install from the inner requirements file.

**Windows PowerShell:**

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r shoe_organizer/requirements.txt
```

**Mac or Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r shoe_organizer/requirements.txt
```

That installs Flask, OpenCV (headless), NumPy, Pillow, PyYAML, python-dotenv, pyserial, onnxruntime, and onnx. Pyserial is required for imports even if you do not use USB serial. Optional Pi-only packages (GPIO, DHT) are commented inside `shoe_organizer/requirements.txt`.

## How to run

With the venv still activated, run from **this repository root** (the folder that contains `run.py` and the `shoe_organizer` subfolder):

```
python run.py
```

Or run from inside `shoe_organizer` after `cd shoe_organizer`:

```
python run.py
```

Then open a browser at:

```
http://localhost:8080
```

Stop the server with Ctrl+C in the terminal.

Default port is 8080. Change `server.port` in `shoe_organizer/config.yaml` if that port is busy.

### Laptop without Pi hardware

To mock motors and GPIO:

**Windows PowerShell:**

```
$env:MOCK_HARDWARE = "1"
python run.py
```

**Mac or Linux:**

```
export MOCK_HARDWARE=1
python run.py
```

## Configuration

Edit `shoe_organizer/config.yaml` for camera index, server host and port, vision options, and optional ONNX shoe detection under `shoe_object_detection` if you add a local `.onnx` model file.
