"""
Fused shoe-type classification: sports | casual | leather.

Combines (1) histogram similarity to datasets/shoe_types/ (mean of top-k refs per type),
(2) OpenCV vision prior from the live frame, (3) shoe style catalog folder names when available.

Tune weights and temperature under config `shoe_type_dataset.fusion`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config_loader import load_config
from .shoe_taxonomy import resolve_shoe_type
from .shoe_type_dataset import compute_type_histogram_scores_from_bgr, match_shoe_type_from_dataset
from .vision_service import ShoeCategory, VisionResult

_ORDER = ("casual", "leather", "sports")


@dataclass
class ShoeTypeClassification:
    shoe_type: str
    confidence: float
    fused_probs: dict[str, float]
    hist_scores: dict[str, float]
    vision_prior: dict[str, float]
    catalog_prior: dict[str, float]
    backend: str


def _vision_prior(category: ShoeCategory) -> dict[str, float]:
    """Soft distribution from edge/saturation heuristics (not a hard label)."""
    if category == ShoeCategory.LEATHER:
        return {"leather": 0.86, "casual": 0.10, "sports": 0.04}
    if category == ShoeCategory.SPORTS:
        return {"sports": 0.76, "casual": 0.17, "leather": 0.07}
    if category == ShoeCategory.CASUAL:
        return {"casual": 0.80, "sports": 0.12, "leather": 0.08}
    return {"casual": 0.50, "sports": 0.22, "leather": 0.28}


def _catalog_prior(
    catalog_category: str | None,
    catalog_style: str | None,
    vision: VisionResult,
) -> dict[str, float]:
    tkey, _ = resolve_shoe_type(catalog_category, catalog_style, vision)
    base = 0.12
    out = {k: base for k in _ORDER}
    out[tkey] = 0.76
    return out


def _normalize_hist_for_fusion(scores: dict[str, float]) -> dict[str, float]:
    """Map raw correl-like scores to [0,1] per type; missing refs → neutral."""
    active = {k: v for k, v in scores.items() if v >= 0.0}
    if not active:
        return {k: 1.0 / 3.0 for k in _ORDER}
    if len(active) == 1:
        only = next(iter(active))
        return {k: 1.0 if k == only else 0.0 for k in _ORDER}
    lo = min(active.values())
    hi = max(active.values())
    rng = hi - lo
    if rng < 1e-8:
        return {k: 1.0 / len(active) if k in active else 0.0 for k in _ORDER}
    out: dict[str, float] = {}
    for k in _ORDER:
        if scores[k] < 0.0:
            out[k] = 1.0 / 3.0
        else:
            out[k] = (scores[k] - lo) / rng
    return out


def _softmax(logits: dict[str, float], temperature: float) -> dict[str, float]:
    t = max(0.05, float(temperature))
    ex = {k: math.exp(v / t) for k, v in logits.items()}
    s = sum(ex.values()) or 1.0
    return {k: ex[k] / s for k in ex}


def classify_shoe_type(
    bgr_preprocessed: np.ndarray,
    cfg: dict | None,
    vision: VisionResult,
    catalog_category: str | None,
    catalog_style: str | None,
) -> ShoeTypeClassification:
    """
    bgr_preprocessed: ROI+CLAHE frame (same as catalog / gate).
    """
    cfg = cfg or load_config()
    block = cfg.get("shoe_type_dataset", {})
    if not bool(block.get("enabled", True)):
        tkey, _ = resolve_shoe_type(catalog_category, catalog_style, vision)
        probs = {k: 0.0 for k in _ORDER}
        probs[tkey] = 1.0
        return ShoeTypeClassification(
            shoe_type=tkey,
            confidence=1.0,
            fused_probs=probs,
            hist_scores={k: -1.0 for k in _ORDER},
            vision_prior=_vision_prior(vision.category),
            catalog_prior=_catalog_prior(catalog_category, catalog_style, vision),
            backend="opencv_catalog_histogram",
        )

    fusion = block.get("fusion") or {}
    use_fusion = bool(fusion.get("enabled", True))

    hist_raw = compute_type_histogram_scores_from_bgr(
        bgr_preprocessed, cfg, already_preprocessed=True
    )
    has_any_ref = any(v >= 0.0 for v in hist_raw.values())

    if not use_fusion or not has_any_ref:
        td = match_shoe_type_from_dataset(
            bgr_preprocessed, cfg, already_preprocessed=True
        )
        if td.matched and td.shoe_type:
            probs = {k: 0.0 for k in _ORDER}
            probs[td.shoe_type] = 1.0
            return ShoeTypeClassification(
                shoe_type=td.shoe_type,
                confidence=1.0,
                fused_probs=probs,
                hist_scores=td.scores_by_type,
                vision_prior=_vision_prior(vision.category),
                catalog_prior=_catalog_prior(catalog_category, catalog_style, vision),
                backend="opencv_type_dataset",
            )
        tkey, _ = resolve_shoe_type(catalog_category, catalog_style, vision)
        probs = {k: 0.0 for k in _ORDER}
        probs[tkey] = 1.0
        return ShoeTypeClassification(
            shoe_type=tkey,
            confidence=1.0,
            fused_probs=probs,
            hist_scores=hist_raw,
            vision_prior=_vision_prior(vision.category),
            catalog_prior=_catalog_prior(catalog_category, catalog_style, vision),
            backend="opencv_catalog_histogram",
        )

    wh = float(fusion.get("weight_histogram", 0.38))
    wv = float(fusion.get("weight_vision", 0.47))
    wc = float(fusion.get("weight_catalog", 0.15))
    # When OpenCV already says sports / leather / casual, trust vision over a misleading histogram
    extra_sv = float(fusion.get("extra_vision_weight_when_sports", 0.10))
    extra_lv = float(fusion.get("extra_vision_weight_when_leather", 0.12))
    extra_cv = float(fusion.get("extra_vision_weight_when_casual", 0.12))
    if vision.category == ShoeCategory.SPORTS and bool(fusion.get("boost_when_vision_sports", True)):
        wv += extra_sv
        wh = max(0.08, wh - extra_sv)
    elif vision.category == ShoeCategory.LEATHER and bool(fusion.get("boost_when_vision_leather", True)):
        wv += extra_lv
        wh = max(0.08, wh - extra_lv)
    elif vision.category == ShoeCategory.CASUAL and bool(fusion.get("boost_when_vision_casual", True)):
        wv += extra_cv
        wh = max(0.08, wh - extra_cv)
    ws = wh + wv + wc
    if ws <= 0:
        ws = 1.0
    wh, wv, wc = wh / ws, wv / ws, wc / ws

    h_norm = _normalize_hist_for_fusion(hist_raw)
    v_dist = _vision_prior(vision.category)
    c_dist = _catalog_prior(catalog_category, catalog_style, vision)

    logits: dict[str, float] = {}
    for k in _ORDER:
        logits[k] = wh * h_norm[k] + wv * v_dist[k] + wc * c_dist[k]

    if vision.category == ShoeCategory.SPORTS:
        logits["sports"] += float(fusion.get("sports_logit_boost", 0.18))
    elif vision.category == ShoeCategory.LEATHER:
        logits["leather"] += float(fusion.get("leather_logit_boost", 0.2))
    elif vision.category == ShoeCategory.CASUAL:
        logits["casual"] += float(fusion.get("casual_logit_boost", 0.2))

    # Histogram refs can look “sports”; when vision says dress/leather, do not let sports win easily.
    if vision.category == ShoeCategory.CASUAL:
        logits["sports"] -= float(fusion.get("penalize_sports_when_vision_casual", 0.12))
    elif vision.category == ShoeCategory.LEATHER:
        logits["sports"] -= float(fusion.get("penalize_sports_when_vision_leather", 0.1))

    temp = float(fusion.get("temperature", 0.5))
    probs = _softmax(logits, temp)
    winner = max(probs, key=probs.get)
    conf = float(probs[winner])

    min_conf = float(fusion.get("min_confidence", 0.0))
    if min_conf > 0 and conf < min_conf:
        tkey, _ = resolve_shoe_type(catalog_category, catalog_style, vision)
        winner = tkey
        conf = 1.0
        probs = {k: 0.0 for k in _ORDER}
        probs[tkey] = 1.0
        return ShoeTypeClassification(
            shoe_type=winner,
            confidence=conf,
            fused_probs={k: round(float(probs[k]), 4) for k in _ORDER},
            hist_scores=hist_raw,
            vision_prior={k: round(float(v_dist[k]), 4) for k in _ORDER},
            catalog_prior={k: round(float(c_dist[k]), 4) for k in _ORDER},
            backend="opencv_catalog_histogram_fallback",
        )

    return ShoeTypeClassification(
        shoe_type=winner,
        confidence=round(conf, 4),
        fused_probs={k: round(float(probs[k]), 4) for k in _ORDER},
        hist_scores=hist_raw,
        vision_prior={k: round(float(v_dist[k]), 4) for k in _ORDER},
        catalog_prior={k: round(float(c_dist[k]), 4) for k in _ORDER},
        backend="opencv_type_fusion",
    )


def classification_to_api_dict(cl: ShoeTypeClassification) -> dict[str, Any]:
    """Extra API fields for debugging / UI (histogram scores kept as shoe_type_dataset_scores in caller)."""
    return {
        "shoe_type_fusion_probs": cl.fused_probs,
        "shoe_type_fusion_confidence": cl.confidence,
        "shoe_type_vision_prior": cl.vision_prior,
        "shoe_type_catalog_prior": cl.catalog_prior,
    }
