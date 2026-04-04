"""
Crop + illumination normalization before shoe gate / catalog / dirt heuristics.
Tune `vision_preprocess` in config.yaml for your booth camera framing.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def apply_vision_preprocess(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    vp = cfg.get("vision_preprocess", {})
    out = bgr
    roi = vp.get("roi")
    if roi and isinstance(roi, dict):
        cropped = _roi_crop(out, roi)
        if cropped is not None and cropped.size > 0:
            out = cropped
    if vp.get("clahe_enabled", True):
        clip = float(vp.get("clahe_clip", 2.0))
        tile = max(2, int(vp.get("clahe_tile", 8)))
        out = _clahe_bgr(out, clip, tile)
    return out


def _roi_crop(bgr: np.ndarray, roi: dict[str, Any]) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    x = int(float(roi.get("x", 0)) * w)
    y = int(float(roi.get("y", 0)) * h)
    rw = int(float(roi.get("w", 1)) * w)
    rh = int(float(roi.get("h", 1)) * h)
    rw = max(2, min(rw, w))
    rh = max(2, min(rh, h))
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    x2 = min(w, x + rw)
    y2 = min(h, y + rh)
    if x2 <= x or y2 <= y:
        return None
    return bgr[y:y2, x:x2].copy()


def _clahe_bgr(bgr: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
