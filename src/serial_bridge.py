"""
Background thread: reads JSON lines from the ESP32 USB-serial port and
feeds each one into the esp32_telemetry store (same data path as the
HTTP POST /api/esp32/telemetry).

Auto-detects CH340/CP210x COM ports on Windows when port is "auto".
"""
from __future__ import annotations

import json
import logging
import threading
import time

import serial
import serial.tools.list_ports

from .esp32_telemetry import update_from_body

log = logging.getLogger(__name__)

_ESP32_USB_KEYWORDS = ("ch340", "cp210", "esp32", "usb-serial", "silicon labs")


def _auto_detect_port() -> str | None:
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(k in desc for k in _ESP32_USB_KEYWORDS):
            return p.device
    return None


class SerialBridge:
    def __init__(self, cfg: dict) -> None:
        block = cfg.get("esp32_telemetry") or {}
        ser_cfg = block.get("serial") or {}
        self._port: str = str(ser_cfg.get("port", "auto"))
        self._baud: int = int(ser_cfg.get("baud", 115200))
        self._enabled: bool = bool(block.get("enabled", False))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not self._enabled:
            log.info("esp32_telemetry disabled — serial bridge not started")
            return
        if self._port == "auto":
            detected = _auto_detect_port()
            if detected:
                self._port = detected
                log.info("auto-detected ESP32 serial port: %s", self._port)
            else:
                log.warning("esp32_telemetry.serial.port is 'auto' but no CH340/CP210x found")
                return
        self._thread = threading.Thread(
            target=self._run,
            name="esp32-serial-bridge",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._read_loop()
                backoff = 1.0
            except serial.SerialException as e:
                log.warning("serial error on %s: %s — retry in %.0fs", self._port, e, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
            except Exception:
                log.exception("serial bridge crash — retry in %.0fs", backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)

    def _read_loop(self) -> None:
        log.info("opening serial %s @ %d baud", self._port, self._baud)
        with serial.Serial(self._port, self._baud, timeout=2) as ser:
            backoff = 1.0
            while not self._stop.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    continue
                try:
                    body = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("serial: bad JSON: %s", line[:120])
                    continue
                if isinstance(body, dict):
                    update_from_body(body)
                    backoff = 1.0
