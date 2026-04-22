from __future__ import annotations

from dataclasses import dataclass

from .config_loader import load_config
from .vision_service import VisionResult


def _normalize_shoe_type(shoe_type: str | None) -> str:
    """Only casual | sports; legacy \"leather\" maps to casual (gentle treatment)."""
    st = (shoe_type or "casual").lower()
    if st == "leather":
        return "casual"
    if st not in ("casual", "sports"):
        return "casual"
    return st


@dataclass
class WashPlan:
    mode: str
    reason: str


def _effective_soil(d_raw: float, shoe_type: str, w: dict) -> float:
    st = _normalize_shoe_type(shoe_type)
    disc = (w.get("texture_discount") or {}).get(st)
    if disc is None:
        disc = float((w.get("texture_discount") or {}).get("casual", 0.06))
    else:
        disc = float(disc)
    return max(0.0, min(1.0, float(d_raw) - disc))


def _decide_wash_rule_based(vision: VisionResult, shoe_type: str, cfg: dict) -> WashPlan:
    """Rule-based dirt_level: only clearly soiled frames get deep wash."""
    w = cfg.get("wash", {})
    dl = (vision.dirt_level or "clean").lower()
    st = _normalize_shoe_type(shoe_type)
    d_raw = float(vision.dirt_score)
    d = _effective_soil(d_raw, st, w)

    if dl == "very_dirty":
        return WashPlan(
            "hard",
            "Heavy soiling detected — deep wash recommended.",
        )

    skip_thr = float(w.get("skip_wash_effective_below", 0.11))
    if bool(w.get("allow_skip_wash_when_clean", True)) and dl == "clean" and d <= skip_thr:
        return WashPlan(
            "none",
            "Shoe appears clean — no wash cycle recommended.",
        )

    if dl == "clean":
        return WashPlan(
            "soft",
            "Shoe appears clean — gentle wash only; no deep clean.",
        )

    if dl == "dirty":
        return WashPlan(
            "soft",
            "Visible soil detected, but not severe — gentle wash recommended.",
        )

    return WashPlan(
        "soft",
        "Light soil signal — gentle wash recommended (no deep clean).",
    )


def decide_wash(vision: VisionResult, shoe_type: str = "casual") -> WashPlan:
    """
    shoe_type: casual | sports (two-way; legacy \"leather\" treated as casual).
    Chooses hard | soft | none from soil-like score + material class.
    """
    cfg = load_config()
    w = cfg.get("wash", {})
    if bool(cfg.get("vision", {}).get("rule_based_pipeline", False)) and vision.dirt_level:
        return _decide_wash_rule_based(vision, shoe_type, cfg)

    hard_thr = float(w.get("hard_if_dirt_above", 0.40))
    sport_push = float(w.get("sports_hard_if_dirt_above", 0.32))
    casual_push = float(w.get("casual_hard_if_dirt_above", 0.34))
    st = _normalize_shoe_type(shoe_type)

    d_raw = float(vision.dirt_score)
    d = _effective_soil(d_raw, st, w)

    skip_thr = float(w.get("skip_wash_effective_below", 0.11))
    if bool(w.get("allow_skip_wash_when_clean", True)) and d <= skip_thr:
        return WashPlan(
            "none",
            f"Very low soil signal ({d:.2f}) — shoe appears clean; no wash cycle recommended.",
        )

    gentle_thr = float(w.get("gentle_only_effective_below", 0.22))
    if d < gentle_thr:
        return WashPlan(
            "soft",
            f"Low soil signal ({d:.2f}) — shoe appears clean; gentle wash only, no deep clean.",
        )

    if d >= hard_thr:
        return WashPlan(
            "hard",
            f"Heavy soil ({d:.2f}) — deep wash recommended for this {st} shoe",
        )

    if st == "sports":
        if d >= sport_push:
            return WashPlan(
                "hard",
                f"Sports shoe — elevated soil signal ({d:.2f}); deeper clean",
            )
        return WashPlan(
            "soft",
            "Sports shoe — light soil; standard gentle wash",
        )
    if st == "casual":
        if d >= casual_push:
            return WashPlan(
                "hard",
                f"Casual shoe — elevated soil signal ({d:.2f}); stronger wash",
            )
        return WashPlan(
            "soft",
            "Casual shoe — gentle wash",
        )
    return WashPlan(
        "soft",
        f"Casual shoe — gentle wash (soil {d:.2f})",
    )


def wash_ui_label(mode: str, shoe_type: str) -> str:
    labels = {"casual": "Casual", "sports": "Sports", "leather": "Leather"}
    st = labels.get(_normalize_shoe_type(shoe_type), "Casual")
    if mode == "hard":
        return f"Deep wash ({st})"
    if mode == "none":
        return f"No wash ({st})"
    return f"Gentle wash ({st})"
