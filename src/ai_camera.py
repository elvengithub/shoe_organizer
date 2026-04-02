"""
OpenCV + optional TFLite + not-shoe gallery: not-shoe vs shoe, then 4-way type + wash.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config_loader import load_config
from .shoe_catalog import match_against_catalog
from .shoe_decision import raw_shoe_acceptance
from .shoe_taxonomy import format_shoe_display_name, resolve_shoe_type
from .vision_preprocess import apply_vision_preprocess
from .vision_service import VisionResult, analyze_frame, encode_jpeg
from .wash_decision import WashPlan, decide_wash, wash_ui_label

log = logging.getLogger(__name__)

NOT_SHOE_MESSAGE = "Not a shoe."
NO_CATALOG_MESSAGE = "(No matching shoe in catalog.)"
STABILIZING_MESSAGE = "Hold steady — confirming shoe…"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _maybe_save_reject(bgr: np.ndarray, cfg: dict, stage: str) -> None:
    cap = cfg.get("debug_capture", {})
    if not bool(cap.get("enabled", False)):
        return
    if not bool(cap.get("save_rejects", True)):
        return
    try:
        root = _project_root() / str(cap.get("path", "datasets/captured_rejects"))
        root.mkdir(parents=True, exist_ok=True)
        blob = encode_jpeg(bgr, quality=int(cap.get("jpeg_quality", 82)))
        if not blob:
            return
        fn = root / f"reject_{stage}_{int(time.time() * 1000)}.jpg"
        fn.write_bytes(blob)
    except Exception as e:
        log.debug("debug capture skip: %s", e)


def _detail_not_shoe(gate_reason: str, reject_stage: str = "gate", dbg: dict[str, Any] | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "is_shoe": False,
        "raw_is_shoe": False,
        "reject_stage": reject_stage,
        "catalog_match": None,
        "object_classification": "not_shoe",
        "message": NOT_SHOE_MESSAGE,
        "shoe_category": None,
        "shoe_type_label": None,
        "dirt_score": None,
        "wash_mode": None,
        "wash_label": None,
        "wash_reason": None,
        "inference_backend": "phased_classifier",
        "gate_reason": gate_reason,
    }
    if dbg:
        for k in ("tflite_p_shoe", "max_not_shoe_score", "anti_face_reason", "skin_pixel_ratio", "haar_face_area_ratio"):
            if k in dbg:
                d[k] = dbg[k]
    return d


def _detail_no_catalog(score: float, gate_reason: str) -> dict[str, Any]:
    return {
        "is_shoe": True,
        "raw_is_shoe": True,
        "reject_stage": None,
        "catalog_match": False,
        "object_classification": "unknown_catalog",
        "message": NO_CATALOG_MESSAGE,
        "shoe_category": None,
        "shoe_type_label": None,
        "dirt_score": None,
        "wash_mode": None,
        "wash_label": None,
        "wash_reason": None,
        "inference_backend": "opencv_catalog_histogram",
        "gate_reason": gate_reason,
        "catalog_category": None,
        "catalog_style": None,
        "catalog_score": round(float(score), 4),
    }


def analyze_shoe_and_wash_from_bgr(bgr: np.ndarray) -> tuple[VisionResult | None, WashPlan | None, dict[str, Any]]:
    cfg = load_config()
    bgr = apply_vision_preprocess(bgr, cfg)
    ok, stage, dbg = raw_shoe_acceptance(bgr, cfg)
    if not ok:
        _maybe_save_reject(bgr, cfg, stage)
        return None, None, _detail_not_shoe(dbg.get("gate_reason", stage), stage, dbg)

    gate_reason = str(dbg.get("gate_reason", "ok"))
    cm = match_against_catalog(bgr, cfg, already_preprocessed=True)
    if not cm.matched:
        return None, None, _detail_no_catalog(cm.score, gate_reason)

    vision = analyze_frame(bgr)
    shoe_type, type_short = resolve_shoe_type(cm.category, cm.style, vision)
    wash = decide_wash(vision, shoe_type)
    shoe_type_label = format_shoe_display_name(
        shoe_type, type_short, cm.category, cm.style
    )
    wash_label = wash_ui_label(wash.mode, shoe_type)

    detail: dict[str, Any] = {
        "is_shoe": True,
        "raw_is_shoe": True,
        "reject_stage": None,
        "catalog_match": True,
        "object_classification": f"shoe_{shoe_type}",
        "shoe_category": shoe_type,
        "shoe_type_label": shoe_type_label,
        "shoe_type_short": type_short,
        "dirt_score": round(float(vision.dirt_score), 4),
        "wash_mode": wash.mode,
        "wash_label": wash_label,
        "wash_reason": wash.reason,
        "inference_backend": "opencv_catalog_histogram",
        "gate_reason": gate_reason,
        "catalog_category": cm.category,
        "catalog_style": cm.style,
        "catalog_score": round(float(cm.score), 4),
    }
    if "tflite_p_shoe" in dbg:
        detail["tflite_p_shoe"] = dbg["tflite_p_shoe"]
    return vision, wash, detail
