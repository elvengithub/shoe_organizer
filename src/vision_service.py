from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .config_loader import load_config

log = logging.getLogger(__name__)


class ShoeCategory(str, Enum):
    LEATHER = "leather"
    SPORTS = "sports"
    UNKNOWN = "unknown"


@dataclass
class VisionResult:
    dirt_score: float
    category: ShoeCategory
    frame_bgr: np.ndarray | None = None


@dataclass
class ShoeGateResult:
    """OpenCV silhouette heuristics — tune in config `shoe_gate`. Not a full classifier."""

    is_shoe: bool
    reason: str


def evaluate_shoe_gate(bgr: np.ndarray, cfg: dict | None = None) -> ShoeGateResult:
    cfg = cfg or load_config()
    sg = cfg.get("shoe_gate", {})
    if not bool(sg.get("enabled", True)):
        return ShoeGateResult(True, "gate_disabled")

    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return ShoeGateResult(False, "frame_too_small")

    area_frame = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ShoeGateResult(False, "no_object")

    cnt = max(contours, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt))
    ratio = a / area_frame
    min_r = float(sg.get("min_contour_area_ratio", 0.025))
    max_r = float(sg.get("max_contour_area_ratio", 0.92))
    if ratio < min_r:
        return ShoeGateResult(False, "object_too_small")
    if ratio > max_r:
        return ShoeGateResult(False, "object_fills_frame")

    _x, _y, rw, rh = cv2.boundingRect(cnt)
    rw = max(rw, 1)
    rh = max(rh, 1)
    ar = max(rw, rh) / min(rw, rh)
    min_ar = float(sg.get("min_aspect_ratio", 1.1))
    max_ar = float(sg.get("max_aspect_ratio", 7.5))
    if ar < min_ar:
        return ShoeGateResult(False, "too_compact")
    if ar > max_ar:
        return ShoeGateResult(False, "too_elongated")

    hull = cv2.convexHull(cnt)
    hull_a = float(cv2.contourArea(hull))
    solidity = a / max(hull_a, 1e-6)
    min_sol = float(sg.get("min_solidity", 0.32))
    if solidity < min_sol:
        return ShoeGateResult(False, "too_irregular")

    return ShoeGateResult(True, "ok")


def _dirt_score(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edge_density = float(np.mean(edges > 0))
    std = float(np.std(gray)) / 255.0
    return min(1.0, 0.5 * edge_density + 0.5 * std)


def _category_heuristic(bgr: np.ndarray) -> ShoeCategory:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = float(np.mean(hsv[:, :, 1]))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_d = float(np.mean(edges > 0))
    cfg = load_config()
    vm = cfg.get("vision", {})
    if s < vm.get("leather_saturation_max", 45) and edge_d < vm.get("sports_edge_min", 0.08) + 0.02:
        return ShoeCategory.LEATHER
    if edge_d >= vm.get("sports_edge_min", 0.08):
        return ShoeCategory.SPORTS
    return ShoeCategory.UNKNOWN


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes() if ok else None


def blank_jpeg_bytes(width: int = 640, height: int = 480) -> bytes:
    img = np.zeros((max(2, height), max(2, width), 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b""


def analyze_frame(bgr: np.ndarray) -> VisionResult:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    d = _dirt_score(gray)
    cat = _category_heuristic(bgr)
    return VisionResult(dirt_score=d, category=cat, frame_bgr=bgr)


class WebcamCapture:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        c = self.cfg["camera"]
        self.index = int(c["index"])
        self.w = int(c["width"])
        self.h = int(c["height"])
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._cap is None:
                self.open()
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.error("webcam read failed")
                return None
            return frame

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
