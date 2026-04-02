"""
Phased shoe vs non-shoe: gate → optional TFLite → optional not-shoe template gallery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .anti_face import reject_if_face_or_skin
from .config_loader import load_config
from .not_shoe_gallery import max_not_shoe_similarity
from .vision_service import evaluate_shoe_gate


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def raw_shoe_acceptance(bgr_preprocessed, cfg: dict | None = None) -> tuple[bool, str, dict[str, Any]]:
    """
    Returns (accept_as_shoe, stage_code, debug_dict).
    stage_code: ok | gate | anti_face | binary | negative_template
    """
    cfg = cfg or load_config()
    dbg: dict[str, Any] = {}

    gate = evaluate_shoe_gate(bgr_preprocessed, cfg)
    dbg["gate_reason"] = gate.reason
    if not gate.is_shoe:
        return False, "gate", dbg

    af_reject, af_reason, af_dbg = reject_if_face_or_skin(bgr_preprocessed, cfg)
    dbg.update(af_dbg)
    if af_reject:
        dbg["anti_face_reason"] = af_reason
        return False, "anti_face", dbg

    sb = cfg.get("shoe_binary", {})
    model_rel = sb.get("model_path", "models/shoe_binary.tflite")
    model_path = _root() / model_rel
    use_binary = bool(sb.get("enabled", False)) and model_path.is_file()

    if use_binary:
        from .shoe_binary_tflite import predict_p_shoe

        p = predict_p_shoe(bgr_preprocessed, str(model_path), sb)
        dbg["tflite_p_shoe"] = p
        if p is None:
            if bool(sb.get("strict", False)):
                return False, "binary", dbg
        else:
            thr = float(sb.get("threshold", 0.5))
            if p < thr:
                return False, "binary", dbg
            if bool(sb.get("skip_negative_gallery_after_binary", True)):
                return True, "ok", dbg

    ns = cfg.get("not_shoe_catalog", {})
    if bool(ns.get("enabled", True)):
        nmax = max_not_shoe_similarity(bgr_preprocessed, cfg)
        dbg["max_not_shoe_score"] = nmax
        thr = float(ns.get("min_score_to_reject", 0.45))
        if nmax >= thr:
            return False, "negative_template", dbg

    return True, "ok", dbg
