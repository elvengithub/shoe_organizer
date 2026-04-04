"""
Reject scenes that look like a person / face (common false positive for "shoe").
Runs after the silhouette gate. User sees the same message as other non-shoes: Not a shoe.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config_loader import load_config

log = logging.getLogger(__name__)


def _load_cascade(rel_name: str) -> cv2.CascadeClassifier | None:
    bundled = Path(cv2.data.haarcascades) / rel_name
    if not bundled.is_file():
        log.warning("Haar cascade missing: %s", bundled)
        return None
    cc = cv2.CascadeClassifier(str(bundled))
    return cc if not cc.empty() else None


def reject_if_face_or_skin(bgr: np.ndarray, cfg: dict | None = None) -> tuple[bool, str, dict[str, Any]]:
    """
    Returns (reject_as_not_shoe, reason_code, debug).
    reason_code: "" if keep, else frontal_face | profile_face | dominant_skin_tone
    """
    cfg = cfg or load_config()
    af = cfg.get("anti_face", {})
    dbg: dict[str, Any] = {}

    if not bool(af.get("enabled", True)):
        return False, "", dbg

    if bgr is None or bgr.size == 0:
        return False, "", dbg

    h, w = bgr.shape[:2]
    area_img = float(max(1, w * h))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    min_ar = float(af.get("min_face_area_ratio", 0.015))
    scale = float(af.get("haar_scale", 1.08))
    neigh = int(af.get("haar_min_neighbors", 5))
    min_side = int(af.get("haar_min_size", max(28, min(w, h) // 14)))

    if bool(af.get("use_haar_frontal", True)):
        cc = _load_cascade(str(af.get("haar_frontal", "haarcascade_frontalface_default.xml")))
        if cc is not None:
            faces = cc.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=neigh,
                minSize=(min_side, min_side),
            )
            for (_x, _y, ww, hh) in faces:
                ar = (ww * hh) / area_img
                dbg["haar_face_area_ratio"] = round(ar, 4)
                if ar >= min_ar:
                    return True, "frontal_face", dbg

    if bool(af.get("use_haar_profile", True)):
        cc2 = _load_cascade(str(af.get("haar_profile", "haarcascade_profileface.xml")))
        if cc2 is not None:
            faces = cc2.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=neigh + 1,
                minSize=(min_side, min_side),
            )
            for (_x, _y, ww, hh) in faces:
                ar = (ww * hh) / area_img
                dbg["haar_profile_area_ratio"] = round(ar, 4)
                if ar >= min_ar:
                    return True, "profile_face", dbg

    if bool(af.get("use_skin_pixel_ratio", True)):
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        lo = np.array(
            [int(af.get("ycrcb_y_min", 0)), int(af.get("ycrcb_cr_min", 135)), int(af.get("ycrcb_cb_min", 85))],
            dtype=np.uint8,
        )
        hi = np.array(
            [
                int(af.get("ycrcb_y_max", 255)),
                int(af.get("ycrcb_cr_max", 180)),
                int(af.get("ycrcb_cb_max", 135)),
            ],
            dtype=np.uint8,
        )
        mask = cv2.inRange(ycrcb, lo, hi)
        ratio = float(np.mean(mask > 0))
        dbg["skin_pixel_ratio"] = round(ratio, 4)
        thr = float(af.get("skin_ratio_reject", 0.52))
        if ratio >= thr:
            return True, "dominant_skin_tone", dbg

    return False, "", dbg
