from __future__ import annotations

from dataclasses import dataclass

from .config_loader import load_config
from .vision_service import VisionResult


@dataclass
class WashPlan:
    mode: str
    reason: str
    # Filled by decide_wash for API/debug: texture-discounted soil vs raw edge/contrast proxy.
    effective_soil: float | None = None
    raw_soil: float | None = None


def _effective_soil(d_raw: float, shoe_type: str, w: dict) -> float:
    """Reduce edge/texture bias: mesh and knit read 'dirty' from Canny even when visually clean."""
    st = (shoe_type or "casual").lower()
    if st not in ("casual", "sports", "leather"):
        st = "casual"
    disc = (w.get("texture_discount") or {}).get(st)
    if disc is None:
        disc = float((w.get("texture_discount") or {}).get("casual", 0.06))
    else:
        disc = float(disc)
    return max(0.0, min(1.0, float(d_raw) - disc))


def decide_wash(vision: VisionResult, shoe_type: str = "casual") -> WashPlan:
    """
    shoe_type: casual | sports | leather (three-way).
    Chooses hard | soft | none from soil-like score + material class.
    Raw dirt_score is mostly edges + contrast; we discount by type so clean shoes rarely get deep wash.
    """
    cfg = load_config()
    w = cfg.get("wash", {})
    dl = getattr(vision, "dirt_level", None)
    if bool(w.get("use_dirt_level_if_available", True)) and isinstance(dl, str) and dl:
        return _decide_wash_rule_based(vision, shoe_type, cfg)

    hard_thr = float(w.get("hard_if_dirt_above", 0.40))
    sport_push = float(w.get("sports_hard_if_dirt_above", 0.32))
    casual_push = float(w.get("casual_hard_if_dirt_above", 0.34))
    st = (shoe_type or "casual").lower()
    if st not in ("casual", "sports", "leather"):
        st = "casual"

    d_raw = float(vision.dirt_score)
    d = _effective_soil(d_raw, st, w)

    skip_thr = float(w.get("skip_wash_effective_below", 0.11))
    if bool(w.get("allow_skip_wash_when_clean", True)) and d <= skip_thr:
        return WashPlan(
            "none",
            f"Very low soil signal ({d:.2f}) — shoe appears clean; no wash cycle recommended.",
            effective_soil=d,
            raw_soil=d_raw,
        )

    gentle_thr = float(w.get("gentle_only_effective_below", 0.22))
    if d < gentle_thr:
        return WashPlan(
            "soft",
            f"Low soil signal ({d:.2f}) — shoe appears clean; gentle wash only, no deep clean.",
            effective_soil=d,
            raw_soil=d_raw,
        )

    if d >= hard_thr:
        return WashPlan(
            "hard",
            f"Heavy soil ({d:.2f}) — deep wash recommended for this {st} shoe",
            effective_soil=d,
            raw_soil=d_raw,
        )

    if st == "leather":
        return WashPlan(
            "soft",
            "Leather — gentle cycle to reduce scuffing and drying",
            effective_soil=d,
            raw_soil=d_raw,
        )
    if st == "sports":
        if d >= sport_push:
            return WashPlan(
                "hard",
                f"Sports shoe — elevated soil signal ({d:.2f}); deeper clean",
                effective_soil=d,
                raw_soil=d_raw,
            )
        return WashPlan(
            "soft",
            "Sports shoe — light soil; standard gentle wash",
            effective_soil=d,
            raw_soil=d_raw,
        )
    if st == "casual":
        if d >= casual_push:
            return WashPlan(
                "hard",
                f"Casual shoe — elevated soil signal ({d:.2f}); stronger wash",
                effective_soil=d,
                raw_soil=d_raw,
            )
        return WashPlan(
            "soft",
            "Casual shoe — gentle wash",
            effective_soil=d,
            raw_soil=d_raw,
        )
    return WashPlan(
        "soft",
        f"Casual shoe — gentle wash (soil {d:.2f})",
        effective_soil=d,
        raw_soil=d_raw,
    )


def _decide_wash_rule_based(vision: VisionResult, shoe_type: str, cfg: dict) -> WashPlan:
    """When vision.dirt_level is set (rule-based pipeline): only clearly soiled → deep wash."""
    w = cfg.get("wash", {})
    dl = (vision.dirt_level or "clean").lower()
    st = (shoe_type or "casual").lower()
    if st not in ("casual", "sports", "leather"):
        st = "casual"
    d_raw = float(vision.dirt_score)
    d = _effective_soil(d_raw, st, w)

    if dl == "very_dirty":
        return WashPlan(
            "hard",
            "Heavy soiling detected — deep wash recommended.",
            effective_soil=d,
            raw_soil=d_raw,
        )

    skip_thr = float(w.get("skip_wash_effective_below", 0.11))
    if bool(w.get("allow_skip_wash_when_clean", True)) and dl == "clean" and d <= skip_thr:
        return WashPlan(
            "none",
            "Shoe appears clean — no wash cycle recommended.",
            effective_soil=d,
            raw_soil=d_raw,
        )

    if dl == "clean":
        return WashPlan(
            "soft",
            "Shoe appears clean — gentle wash only; no deep clean.",
            effective_soil=d,
            raw_soil=d_raw,
        )

    if dl == "dirty":
        return WashPlan(
            "soft",
            "Visible soiling detected, but not severe — gentle wash recommended.",
            effective_soil=d,
            raw_soil=d_raw,
        )

    # moderate (texture can read as soil; avoid deep wash unless very_dirty above)
    return WashPlan(
        "soft",
        "Light soil signal — gentle wash recommended (no deep clean).",
        effective_soil=d,
        raw_soil=d_raw,
    )


def wash_ui_label(mode: str, shoe_type: str) -> str:
    labels = {"casual": "Casual", "sports": "Sports", "leather": "Leather"}
    st = labels.get((shoe_type or "casual").lower(), "Casual")
    if mode == "hard":
        return f"Deep wash ({st})"
    if mode == "none":
        return f"No wash ({st})"
    return f"Gentle wash ({st})"
