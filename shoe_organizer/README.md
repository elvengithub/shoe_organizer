# Shoe organizer

Flask web app for a shoe cleaning bay: live camera view, shoe vs non-shoe checks, type hints (sports / casual / leather), wash suggestions, and storage slot status. Runs on a PC for development or on a Raspberry Pi next to the hardware.

---

## What you need first

| Requirement | Notes |
|-------------|--------|
| **Python 3.10+** (3.11+ recommended) | [python.org/downloads](https://www.python.org/downloads/) — on Windows, enable **Add python.exe to PATH**. |
| **USB webcam** (optional for dev) | Or ESP32 sending JPEGs — see `config.yaml` → `camera`. |
| **Terminal** | PowerShell, Command Prompt, or Terminal on Mac/Linux. |

---

## Dependencies (install these so the app runs smoothly)

All of the following are declared in **`requirements.txt`**. Install them **once per project** with `pip install -r requirements.txt` (inside your virtual environment).

| Package | Why it’s needed |
|---------|------------------|
| **Flask** (≥3.0) | Web server and HTML UI. |
| **opencv-python-headless** (≥4.8) | Camera frames, image preprocessing, shoe / not-shoe heuristics. |
| **numpy** (≥1.24) | Arrays for OpenCV. |
| **Pillow** (≥10) | Loading catalog / dataset images (e.g. PNG, JPEG). |
| **PyYAML** (≥6) | Reads **`config.yaml`**. |
| **python-dotenv** (≥1) | Optional `.env` overrides. |
| **pyserial** (≥3.5) | USB serial for ESP32 telemetry (`serial_bridge`). Required for imports even if you don’t use a serial device. |

**Optional (not in `requirements.txt` by default):**

- **`tflite-runtime`** — on-device shoe binary model (see comments in `requirements.txt`).
- **`RPi.GPIO`**, **`adafruit-circuitpython-dht`** — real GPIO / DHT on Raspberry Pi (see comments in `requirements.txt`).

---

## How to run the project (new user)

### 1. Get the code

Clone or download the repo, then open a terminal in the **project root** — the folder that contains **`run.py`**, **`requirements.txt`**, **`config.yaml`**, and the **`src`** folder.

```bash
git clone https://github.com/elvengithub/shoe_organizer.git
cd shoe_organizer
```

### 2. Create a virtual environment (recommended)

Keeps packages isolated from other Python projects.

**Windows (PowerShell):**

```text
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**

```text
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of the prompt.

### 3. Install dependencies

```text
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Wait until everything finishes without errors.

### 4. Start the app

```text
python run.py
```

### 5. Open the UI

In a browser, go to:

**http://localhost:8080**

(To stop the server: focus the terminal and press **Ctrl+C**.)

---

## Run on a laptop without motors / GPIO

If you don’t have Raspberry Pi hardware connected, set mock mode **before** `python run.py`:

**PowerShell:**

```text
$env:MOCK_HARDWARE = "1"
python run.py
```

**Mac / Linux:**

```text
export MOCK_HARDWARE=1
python run.py
```

---

## If something goes wrong

| Problem | What to try |
|---------|-------------|
| **`python` is not recognized** | Reinstall Python with **Add to PATH**, or use `py -m venv .venv` on Windows. |
| **`pip install` can’t find `requirements.txt`** | `cd` to the folder that contains **`requirements.txt`** (same folder as `run.py`). |
| **Browser can’t open the page** | Confirm `python run.py` is still running and use **http://localhost:8080**. |
| **Port 8080 in use** | Change `server.port` under **`server:`** in **`config.yaml`**, then restart. |
| **No camera image** | Check `camera.index` / `camera.source` in **`config.yaml`**; on Windows try index `0` or `1`. |

---

## Useful paths

| Path | Purpose |
|------|---------|
| `config.yaml` | Camera, GPIO, vision thresholds, catalogs. |
| `datasets/shoe_types/` | Reference photos per type: `sports`, `casual`, `leather`. |
| `datasets/shoes/` | Style catalog (if you use histogram matching). |
| `CHECKPOINT_4_04_PM.md` | Notes for a saved feature snapshot. |

---

## How to push your code to GitHub

You need **[Git](https://git-scm.com/downloads)** installed and a **[GitHub](https://github.com)** account. The upstream repo is:

**https://github.com/elvengithub/shoe_organizer**

### Use the correct folder (important)

Git tracks this project from the directory that contains **`src/`**, **`config.yaml`**, **`run.py`**, and a **`.git`** folder — all in the **same** place.

- If you only see `run.py` and `requirements.txt` **and** another nested folder also named `shoe_organizer`, that outer folder is **not** the app repo. **`cd` into the inner folder** (the one with `src\`) before running `git status` or `git push`.
- Run `git status` from the folder where `git log` works; that is the repo you push.

### You already cloned the project (most common)

1. Open a terminal in the **repository root** (the folder that contains **`src/`** and **`.git`**).
2. Check changes:

   ```bash
   git status
   ```

3. Commit and push:

   ```bash
   git add -A
   git commit -m "Describe your change in one short line"
   git push origin main
   ```

   If your default branch is named `master`, use `git push origin master` instead.

4. If **`git push` is rejected** because the remote has commits you do not have locally, pull first, then push:

   ```bash
   git pull origin main --rebase
   git push origin main
   ```

### Add the GitHub remote (first time or wrong remote)

`git remote add` always needs **two** arguments: a **name** (usually `origin`) and the **URL**.

```bash
git remote add origin https://github.com/elvengithub/shoe_organizer.git
```

List remotes: `git remote -v`. If `origin` already points somewhere else, update it:

```bash
git remote set-url origin https://github.com/elvengithub/shoe_organizer.git
```

Never run `git remote add` with only the URL — Git will show a usage error.

### First time turning a local folder into this GitHub repo

If the project folder is **not** a Git repo yet (no `.git`):

```bash
cd /path/to/shoe_organizer
git init
git branch -M main
git remote add origin https://github.com/elvengithub/shoe_organizer.git
git add -A
git commit -m "Initial commit"
git push -u origin main
```

If the **remote already has history** and push is rejected, either integrate with `git pull origin main --rebase` (and resolve conflicts), or — only if you intend to **replace** the remote history — discuss `git push --force-with-lease` with your team (dangerous on shared repos).

### Signing in to GitHub (HTTPS)

GitHub does **not** accept your account password for `git push` over HTTPS. Use one of these:

- **GitHub Desktop** or **GitHub CLI** (`gh auth login`), or  
- A **Personal Access Token** as the password when Git asks (create one under GitHub → **Settings → Developer settings → Personal access tokens**), or  
- **SSH**: add an SSH key to GitHub and use the SSH remote URL instead of `https://...`.

### What should not be pushed

- **`.venv/`** — virtual environment (already listed in `.gitignore`).
- **`.env`** — secrets (ignored if present).
- Large private datasets or API keys — keep them out of the repo or use a private remote.

---

## Repository

**https://github.com/elvengithub/shoe_organizer**
