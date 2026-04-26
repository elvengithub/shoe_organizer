"""
OpenCV camera capture with optional frame size normalization (two-stage pipeline).
Uses the same `camera` block as the Flask stack; normalize size from `ai_pipeline.capture_normalize`.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from .config_loader import load_config

log = logging.getLogger(__name__)


class NormalizedCamera:
    """Video capture that resizes every frame to a fixed (width, height)."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        c = self.cfg["camera"]
        pipe = self.cfg.get("ai_pipeline") or {}
        norm = pipe.get("capture_normalize")
        if norm and len(norm) >= 2:
            self.norm_w = int(norm[0])
            self.norm_h = int(norm[1])
        else:
            self.norm_w = int(c["width"])
            self.norm_h = int(c["height"])
        self.index = int(c["index"])
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.norm_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.norm_h)

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._cap is None:
                self.open()
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.error("NormalizedCamera read failed")
                return None
            h, w = frame.shape[:2]
            if w != self.norm_w or h != self.norm_h:
                frame = cv2.resize(frame, (self.norm_w, self.norm_h), interpolation=cv2.INTER_AREA)
            return frame

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
