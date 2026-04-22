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
from .shoe_taxonomy import SHOE_TYPE_LABELS, format_shoe_display_name
from .shoe_type_classifier import classification_to_api_dict, classify_shoe_type
from .vision_preprocess import apply_vision_preprocess
from .vision_service import VisionResult, analyze_frame, encode_jpeg
from .wash_decision import WashPlan, decide_wash, wash_ui_label

log = logging.getLogger(__name__)

NOT_SHOE_MESSAGE = "Not a shoe."
NO_CATALOG_MESSAGE = "(No matching shoe in catalog.)"
STABILIZING_MESSAGE = "Hold steady — confirming shoe…"

# Stable codes for clients / error trapping (not_shoe path)
NOT_SHOE_STAGE_CODES: dict[str, str] = {
    "gate": "NOT_SHOE_GATE",
    "bland_scene": "NOT_SHOE_BLAND_SCENE",
    "anti_face": "NOT_SHOE_FACE_OR_SKIN",
    "binary": "NOT_SHOE_MODEL_REJECT",
    "negative_template": "NOT_SHOE_TEMPLATE_MATCH",
}

NOT_SHOE_STAGE_HINTS: dict[str, str] = {
    "gate": "No shoe-like object detected — place a shoe in the cleaning bay.",
    "bland_scene": "No shoe in view — the bay looks empty or too uniform; place footwear in the cleaning bay.",
    "anti_face": "Looks like a face or skin; aim the camera at footwear only.",
    "binary": "Shoe classifier scored this as not a shoe.",
    "negative_template": "Too similar to saved non-shoe examples.",
}

PIPELINE_ERROR_CODE = "ANALYSIS_PIPELINE_ERROR"


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
    code = NOT_SHOE_STAGE_CODES.get(reject_stage, "NOT_SHOE_REJECTED")
    hint = NOT_SHOE_STAGE_HINTS.get(reject_stage, "")
    d: dict[str, Any] = {
        "is_shoe": False,
        "raw_is_shoe": False,
        "reject_stage": reject_stage,
        "catalog_match": None,
        "object_classification": "not_shoe",
        "message": NOT_SHOE_MESSAGE,
        "classification_error": code,
        "reject_detail": hint,
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
        for k in (
            "tflite_p_shoe",
            "max_not_shoe_score",
            "anti_face_reason",
            "skin_pixel_ratio",
            "haar_face_area_ratio",
            "scene_laplacian_variance",
            "scene_edge_density",
            "scene_gray_std_norm",
            "bland_scene_reason",
        ):
            if k in dbg:
                d[k] = dbg[k]
    return d


def _detail_pipeline_error(exc: BaseException) -> dict[str, Any]:
    """Unexpected failure inside vision — logged, safe message for API."""
    return {
        "is_shoe": False,
        "raw_is_shoe": False,
        "reject_stage": "pipeline_error",
        "catalog_match": None,
        "object_classification": "error",
        "message": "Could not analyze the camera image. Check the camera or try again.",
        "classification_error": PIPELINE_ERROR_CODE,
        "reject_detail": "",
        "shoe_category": None,
        "shoe_type_label": None,
        "dirt_score": None,
        "wash_mode": None,
        "wash_label": None,
        "wash_reason": None,
        "inference_backend": "error",
        "gate_reason": "pipeline_error",
    }


def _detail_no_catalog(score: float, gate_reason: str) -> dict[str, Any]:
    return {
        "is_shoe": True,
        "raw_is_shoe": True,
        "reject_stage": None,
        "catalog_match": False,
        "object_classification": "unknown_catalog",
        "message": NO_CATALOG_MESSAGE,
        "classification_error": "NO_CATALOG_MATCH",
        "reject_detail": "Add similar shoes to the catalog or adjust thresholds in config.",
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


def _analyze_shoe_and_wash_from_bgr_impl(bgr: np.ndarray) -> tuple[VisionResult | None, WashPlan | None, dict[str, Any]]:
    if bgr is None or not hasattr(bgr, "shape") or bgr.size == 0 or len(bgr.shape) < 2:
        return None, None, _detail_not_shoe("invalid_frame", "gate", {"gate_reason": "invalid_frame"})

    cfg = load_config()
    # Dirt score: ROI crop only — CLAHE sharpens edges and raises gray std, which falsely reads as soil.
    bgr_dirt = apply_vision_preprocess(bgr, cfg, include_clahe=False)
    bgr = apply_vision_preprocess(bgr, cfg)
    ok, stage, dbg = raw_shoe_acceptance(bgr, cfg)
    if not ok:
        _maybe_save_reject(bgr, cfg, stage)
        gr = dbg.get("bland_scene_reason") or dbg.get("gate_reason") or stage
        return None, None, _detail_not_shoe(str(gr), stage, dbg)

    gate_reason = str(dbg.get("gate_reason", "ok"))
    cm = match_against_catalog(bgr, cfg, already_preprocessed=True)
    if not cm.matched:
        return None, None, _detail_no_catalog(cm.score, gate_reason)

    vision = analyze_frame(bgr, bgr_for_dirt=bgr_dirt)
    tcls = classify_shoe_type(bgr, cfg, vision, cm.category, cm.style)
    shoe_type = tcls.shoe_type
    type_short = SHOE_TYPE_LABELS[shoe_type]
    type_backend = tcls.backend
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
        "classification_error": None,
        "reject_detail": "",
        "shoe_category": shoe_type,
        "shoe_type_label": shoe_type_label,
        "shoe_type_short": type_short,
        "dirt_score": round(float(vision.dirt_score), 4),
        "dirt_level": vision.dirt_level,
        "dirty_region_ratio": (
            round(float(vision.dirty_region_ratio), 5) if vision.dirty_region_ratio is not None else None
        ),
        "wash_mode": wash.mode,
        "wash_label": wash_label,
        "wash_reason": wash.reason,
        "wash_effective_soil": (
            round(float(wash.effective_soil), 4) if wash.effective_soil is not None else None
        ),
        "wash_raw_soil": round(float(wash.raw_soil), 4) if wash.raw_soil is not None else None,
        "inference_backend": type_backend,
        "gate_reason": gate_reason,
        "catalog_category": cm.category,
        "catalog_style": cm.style,
        "catalog_score": round(float(cm.score), 4),
        **classification_to_api_dict(tcls),
        "shoe_type_dataset_matched": any(v >= 0.0 for v in tcls.hist_scores.values()),
        "shoe_type_dataset_score": (
            round(float(tcls.hist_scores[shoe_type]), 4)
            if tcls.hist_scores.get(shoe_type, -1.0) >= 0.0
            else None
        ),
        "shoe_type_dataset_scores": {
            k: round(float(v), 4) for k, v in tcls.hist_scores.items()
        },
    }
    if "tflite_p_shoe" in dbg:
        detail["tflite_p_shoe"] = dbg["tflite_p_shoe"]
    return vision, wash, detail


def infer_shoe_type_from_bgr_raw(bgr: np.ndarray, cfg: dict | None = None) -> tuple[str | None, np.ndarray | None]:
    """
    After shoe gate + ROI/CLAHE preprocess, infer sports/casual/leather without catalog match.
    Used to label auto-captured dataset images when the identity histogram does not match yet.
    """
    cfg = cfg or load_config()
    bgr_dirt = apply_vision_preprocess(bgr, cfg, include_clahe=False)
    bgr_p = apply_vision_preprocess(bgr, cfg)
    ok, _stage, _dbg = raw_shoe_acceptance(bgr_p, cfg)
    if not ok:
        return None, None
    vision = analyze_frame(bgr_p, bgr_for_dirt=bgr_dirt)
    tcls = classify_shoe_type(bgr_p, cfg, vision, None, None)
    return tcls.shoe_type, bgr_p


def analyze_shoe_and_wash_from_bgr(bgr: np.ndarray) -> tuple[VisionResult | None, WashPlan | None, dict[str, Any]]:
    """
    Classify the object in frame: not-shoe (with classification_error code) or shoe type + wash.
    Exceptions are trapped and returned as pipeline_error detail (never raised).
    """
    try:
        return _analyze_shoe_and_wash_from_bgr_impl(bgr)
    except Exception as e:
        log.exception("shoe vision pipeline failed")
        return None, None, _detail_pipeline_error(e)
