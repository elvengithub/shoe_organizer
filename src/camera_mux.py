"""
Camera input: USB webcam and/or JPEG frames posted from an ESP32 (MicroPython).
See config.yaml `camera.source`: usb | esp32 | prefer_esp32
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .vision_service import WebcamCapture

log = logging.getLogger(__name__)


class CameraMux:
    """Routes `read()` to USB, latest ESP32 frame, or ESP-first with USB fallback."""

    def __init__(self, cfg: dict, usb: WebcamCapture) -> None:
        self.cfg = cfg
        self.usb = usb
        self._esp_lock = threading.Lock()
        self._esp_bgr: np.ndarray | None = None
        self._esp_ts: float = 0.0

    def _esp_fresh(self) -> tuple[np.ndarray | None, bool]:
        c = self.cfg.get("camera", {})
        ttl = float(c.get("esp32_frame_ttl_seconds", 3.0))
        now = time.time()
        with self._esp_lock:
            if self._esp_bgr is None:
                return None, False
            if (now - self._esp_ts) > ttl:
                return None, False
            return self._esp_bgr.copy(), True

    def ingest_jpeg(self, jpeg_bytes: bytes) -> tuple[bool, str]:
        if not jpeg_bytes:
            return False, "empty_body"
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            return False, "invalid_jpeg"
        with self._esp_lock:
            self._esp_bgr = im
            self._esp_ts = time.time()
        return True, "ok"

    def read(self) -> np.ndarray | None:
        c = self.cfg.get("camera", {})
        mode = str(c.get("source", "usb")).lower().strip()
        esp_frame, esp_ok = self._esp_fresh()

        if mode == "esp32":
            return esp_frame if esp_ok else None
        if mode == "prefer_esp32":
            if esp_ok and esp_frame is not None:
                return esp_frame
            return self.usb.read()
        # usb (default)
        return self.usb.read()

    def release(self) -> None:
        self.usb.release()
        with self._esp_lock:
            self._esp_bgr = None
            self._esp_ts = 0.0
