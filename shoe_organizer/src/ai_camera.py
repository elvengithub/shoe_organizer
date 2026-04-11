"""
OpenCV + optional TFLite + not-shoe gallery: not-shoe vs shoe, then 4-way type + wash.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config_loader import load_config
from .shoe_catalog import match_against_catalog
from .shoe_decision import raw_shoe_acceptance
from .shoe_taxonomy import SHOE_TYPE_LABELS, format_shoe_display_name
from .shoe_type_classifier import classification_to_api_dict, classify_shoe_type
from .shoe_type_dataset import match_shoe_type_from_dataset
from .vision_preprocess import apply_vision_preprocess
from .vision_service import ShoeCategory, VisionResult, analyze_frame, encode_jpeg
from .wash_decision import WashPlan, decide_wash, wash_ui_label

log = logging.getLogger(__name__)

NOT_SHOE_MESSAGE = "Not a shoe."
NO_CATALOG_MESSAGE = "(No matching shoe in catalog.)"
STABILIZING_MESSAGE = "Hold steady — confirming shoe…"

# Stable codes for clients / error trapping (not_shoe path)
NOT_SHOE_STAGE_CODES: dict[str, str] = {
    "gate": "NOT_SHOE_GATE",
    "anti_face": "NOT_SHOE_FACE_OR_SKIN",
    "binary": "NOT_SHOE_MODEL_REJECT",
    "negative_template": "NOT_SHOE_TEMPLATE_MATCH",
}

NOT_SHOE_STAGE_HINTS: dict[str, str] = {
    "gate": "No shoe-like object detected — place a shoe in the cleaning bay.",
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
        for k in ("tflite_p_shoe", "max_not_shoe_score", "anti_face_reason", "skin_pixel_ratio", "haar_face_area_ratio"):
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
    bgr = apply_vision_preprocess(bgr, cfg)
    ok, stage, dbg = raw_shoe_acceptance(bgr, cfg)
    if not ok:
        _maybe_save_reject(bgr, cfg, stage)
        return None, None, _detail_not_shoe(dbg.get("gate_reason", stage), stage, dbg)

    gate_reason = str(dbg.get("gate_reason", "ok"))

    if bool(cfg.get("vision", {}).get("rule_based_pipeline", False)):
        vision = analyze_frame(bgr, cfg)
        std = cfg.get("shoe_type_dataset", {}) or {}
        td_match = None
        if bool(std.get("enabled", True)):
            try:
                td_match = match_shoe_type_from_dataset(bgr, cfg, already_preprocessed=True)
            except Exception as e:
                log.warning("shoe_type_dataset match failed (using vision only): %s", e)
                td_match = None

        # Dataset histogram match is the PRIMARY classifier (most accurate).
        # Vision rule-based fusion is the FALLBACK when dataset has no confident match.
        if td_match is not None and td_match.matched and td_match.shoe_type:
            shoe_type = td_match.shoe_type
            vision = replace(
                vision,
                category=ShoeCategory.SPORTS if shoe_type == "sports" else ShoeCategory.CASUAL,
            )
            log.info("shoe type from DATASET match: %s (score=%.3f, scores=%s)",
                     shoe_type, td_match.score, td_match.scores_by_type)
        else:
            shoe_type = vision.category.value
            log.info("shoe type from VISION rules (no dataset match): %s (fusion=%.3f)",
                     shoe_type, vision.sports_fusion_score or 0.0)
        wash = decide_wash(vision, shoe_type)
        type_short = SHOE_TYPE_LABELS[shoe_type]
        shoe_type_label = format_shoe_display_name(shoe_type, type_short, None, None)
        wash_label = wash_ui_label(wash.mode, shoe_type)
        vm = cfg.get("vision", {})
        thr_edge = float(vm.get("sports_edge_density_min", 0.09))
        backend = "opencv_rules"
        if td_match is not None:
            if td_match.matched:
                backend = "opencv_rules_plus_shoe_types_dataset"
            elif any(v >= 0.0 for v in td_match.scores_by_type.values()):
                backend = "opencv_rules_dataset_no_confident_match"
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
            "edge_density": round(float(vision.edge_density), 4) if vision.edge_density is not None else None,
            "sports_edge_density_min": round(thr_edge, 4),
            "sports_fusion_score": round(float(vision.sports_fusion_score), 4)
            if vision.sports_fusion_score is not None
            else None,
            "sports_fusion_threshold": round(float(vision.sports_fusion_threshold), 4)
            if vision.sports_fusion_threshold is not None
            else None,
            "gradient_mean": round(float(vision.gradient_mean), 4) if vision.gradient_mean is not None else None,
            "texture_rms": round(float(vision.texture_rms), 4) if vision.texture_rms is not None else None,
            "saturation_mean": round(float(vision.saturation_mean), 2) if vision.saturation_mean is not None else None,
            "dirty_pixel_ratio": round(float(vision.dirty_pixel_ratio), 4) if vision.dirty_pixel_ratio is not None else None,
            "wash_mode": wash.mode,
            "wash_label": wash_label,
            "wash_reason": wash.reason,
            "inference_backend": backend,
            "gate_reason": gate_reason,
            "catalog_category": None,
            "catalog_style": None,
            "catalog_score": None,
            "shoe_type_dataset_matched": bool(td_match.matched) if td_match is not None else False,
            "shoe_type_dataset_score": (
                round(float(td_match.score), 4)
                if td_match is not None and td_match.matched
                else None
            ),
            "shoe_type_dataset_scores": (
                {k: round(float(v), 4) for k, v in td_match.scores_by_type.items()}
                if td_match is not None
                else {}
            ),
            "leather_like_casual": bool(vision.leather_like_casual)
            if vision.leather_like_casual is not None
            else None,
        }
        if "tflite_p_shoe" in dbg:
            detail["tflite_p_shoe"] = dbg["tflite_p_shoe"]
        return vision, wash, detail

    cm = match_against_catalog(bgr, cfg, already_preprocessed=True)
    if not cm.matched:
        return None, None, _detail_no_catalog(cm.score, gate_reason)

    vision = analyze_frame(bgr)
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
        "wash_mode": wash.mode,
        "wash_label": wash_label,
        "wash_reason": wash.reason,
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
