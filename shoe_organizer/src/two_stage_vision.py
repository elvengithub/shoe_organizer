"""
YOLOv8 detection + two separate classifier heads → VisionResult + debug payload for the API.
Lazy-loaded singletons avoid reloading weights every frame.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .classifier import DualHeadClassifiers
from .detector import ShoeDetector
from .preprocess import normalize_lighting_bgr
from .vision_service import ShoeCategory, VisionResult

log = logging.getLogger(__name__)

_runtime_detector: ShoeDetector | None = None
_runtime_dual: DualHeadClassifiers | None = None
_runtime_sig: str | None = None


def _app_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_model_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (_app_root() / p).resolve()


def _runtime_signature(cfg: dict) -> str:
    ap = cfg.get("ai_pipeline") or {}
    d = ap.get("detector") or {}
    c = ap.get("classifiers") or {}
    return "|".join(
        [
            str(bool(ap.get("prefer_full_roi", False))),
            str(d.get("weights", "")),
            str(c.get("type_weights", "")),
            str(c.get("cleanliness_weights", "")),
            str(c.get("backbone", "")),
            str(c.get("input_size", "")),
            str(c.get("device", "")),
        ]
    )


def _ensure_runtime(cfg: dict) -> tuple[ShoeDetector | None, DualHeadClassifiers]:
    global _runtime_detector, _runtime_dual, _runtime_sig
    sig = _runtime_signature(cfg)
    if _runtime_dual is not None and sig == _runtime_sig:
        return _runtime_detector, _runtime_dual

    ap = cfg.get("ai_pipeline") or {}
    dcfg = ap.get("detector") or {}
    ccfg = ap.get("classifiers") or {}

    tw = _resolve_model_path(str(ccfg.get("type_weights", "")))
    cw = _resolve_model_path(str(ccfg.get("cleanliness_weights", "")))
    if tw is None or not tw.is_file() or cw is None or not cw.is_file():
        raise FileNotFoundError("Classifier checkpoint(s) missing (type or cleanliness).")

    dual = DualHeadClassifiers(
        tw,
        cw,
        list(ap.get("type_class_names") or []),
        list(ap.get("cleanliness_class_names") or []),
        str(ccfg.get("backbone", "mobilenet_v3_small")),
        pretrained_backbone=bool(ccfg.get("pretrained_backbone", True)),
        input_size=int(ccfg.get("input_size", 224)),
        device_pref=str(ccfg.get("device", "auto")),
    )

    detector: ShoeDetector | None = None
    if not bool(ap.get("prefer_full_roi", False)):
        dw = _resolve_model_path(str(dcfg.get("weights", "")))
        if dw is None or not dw.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {dcfg.get('weights')}")
        detector = ShoeDetector(
            dw,
            confidence=float(dcfg.get("confidence", 0.35)),
            iou=float(dcfg.get("iou", 0.45)),
            imgsz=int(dcfg.get("imgsz", 416)),
        )

    _runtime_detector = detector
    _runtime_dual = dual
    _runtime_sig = sig
    return _runtime_detector, _runtime_dual


def _dirt_level_from_dirty_prob(dirty_prob: float, ap: dict) -> str:
    cl = ap.get("cleanliness") or {}
    t0 = float(cl.get("clean_if_dirty_prob_below", 0.35))
    t1 = float(cl.get("moderate_if_dirty_prob_below", 0.55))
    t2 = float(cl.get("dirty_if_dirty_prob_below", 0.75))
    if dirty_prob < t0:
        return "clean"
    if dirty_prob < t1:
        return "moderate"
    if dirty_prob < t2:
        return "dirty"
    return "very_dirty"


def infer_two_stage(preprocessed_bgr: np.ndarray, cfg: dict) -> tuple[VisionResult | None, dict[str, Any]]:
    """
    Run stage-1 YOLO on ``preprocessed_bgr`` (ROI + CLAHE), then stage-2 classifiers on the crop.
    Returns (None, {}) if disabled, missing weights, or inference error (caller falls back).
    """
    ap = cfg.get("ai_pipeline") or {}
    if not bool(ap.get("enabled")):
        return None, {}

    dbg: dict[str, Any] = {"ai_pipeline": True}
    try:
        detector, dual = _ensure_runtime(cfg)
    except FileNotFoundError as e:
        log.warning("ai_pipeline: %s — using legacy vision.", e)
        return None, {"ai_pipeline_error": str(e)}
    except ImportError as e:
        log.warning("ai_pipeline: missing dependency %s — install requirements-ai.txt", e)
        return None, {"ai_pipeline_error": str(e)}

    t0 = time.perf_counter()
    dcfg = ap.get("detector") or {}
    if detector is None or bool(ap.get("prefer_full_roi", False)):
        crop = preprocessed_bgr
        box = None
        dconf = None
        dbg["prefer_full_roi"] = True
    else:
        det = detector.detect_best(preprocessed_bgr)
        use_full = bool(dcfg.get("use_full_frame_if_no_detection", True))
        if det is None and not use_full:
            dbg["detector_miss"] = True
            return None, dbg

        if det is None:
            crop = preprocessed_bgr
            box = None
            dconf = None
            dbg["detector_miss"] = True
        else:
            crop = det.crop_bgr
            box = det.xyxy
            dconf = det.confidence
            dbg["detector_confidence"] = round(float(dconf), 4)

    if bool(ap.get("normalize_crop_clahe", False)):
        crop = normalize_lighting_bgr(crop)

    try:
        (fine_type, type_conf, type_probs), (clean_lab, clean_conf, clean_probs) = dual.predict_both(crop)
    except Exception as e:
        log.exception("Classifier forward failed")
        return None, {**dbg, "ai_pipeline_error": str(e)}

    dirty_prob = float(clean_probs.get("dirty", 0.0))
    if "dirty" not in clean_probs and "clean" in clean_probs:
        dirty_prob = 1.0 - float(clean_probs["clean"])

    bucket_map = ap.get("type_to_bucket") or {}
    default_b = str(ap.get("type_to_bucket_default", "casual")).lower()
    if default_b not in ("sports", "casual"):
        default_b = "casual"
    bucket = str(bucket_map.get(fine_type, default_b)).lower()
    if bucket not in ("sports", "casual"):
        bucket = "casual"

    dirt_level = _dirt_level_from_dirty_prob(dirty_prob, ap)
    category = ShoeCategory.SPORTS if bucket == "sports" else ShoeCategory.CASUAL

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    dbg.update(
        {
            "fine_shoe_type": fine_type,
            "shoe_type_confidence": round(float(type_conf), 4),
            "cleanliness_label": clean_lab,
            "cleanliness_confidence": round(float(clean_conf), 4),
            "cleanliness_probs": {k: round(float(v), 4) for k, v in clean_probs.items()},
            "shoe_type_probs": {k: round(float(v), 4) for k, v in type_probs.items()},
            "shoe_category_bucket": bucket,
            "dirty_prob": round(float(dirty_prob), 4),
            "inference_ms": round(float(elapsed_ms), 2),
        }
    )
    if bool((ap.get("logging") or {}).get("log_inference_ms")):
        log.info("two_stage inference %.1f ms type=%s clean=%s", elapsed_ms, fine_type, clean_lab)

    vision = VisionResult(
        dirt_score=float(dirty_prob),
        category=category,
        frame_bgr=preprocessed_bgr,
        fine_shoe_type=str(fine_type),
        shoe_type_confidence=float(type_conf),
        cleanliness_confidence=float(clean_conf),
        detector_box_xyxy=tuple(float(x) for x in box) if box is not None else None,
        detector_confidence=float(dconf) if dconf is not None else None,
        dirt_level=dirt_level,
        dirty_pixel_ratio=float(dirty_prob),
    )
    return vision, dbg


def reset_runtime_cache() -> None:
    """Test hook / hot-reload support."""
    global _runtime_detector, _runtime_dual, _runtime_sig
    _runtime_detector = None
    _runtime_dual = None
    _runtime_sig = None
