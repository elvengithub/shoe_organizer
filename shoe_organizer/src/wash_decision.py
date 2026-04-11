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


def _decide_wash_rule_based(vision: VisionResult, shoe_type: str, cfg: dict) -> WashPlan:
    """User spec: very_dirty → deep (hard); sports + moderate → hard; else soft."""
    dl = (vision.dirt_level or "clean").lower()
    st = _normalize_shoe_type(shoe_type)
    if dl == "very_dirty":
        return WashPlan(
            "hard",
            "Deep clean - very dirty (edge + color rules).",
        )
    if st == "sports" and dl == "moderate":
        return WashPlan(
            "hard",
            "Hard wash - sports shoe with moderate soil (rule-based).",
        )
    return WashPlan(
        "soft",
        "Soft wash - light soil or material-safe cycle (rule-based).",
    )


def decide_wash(vision: VisionResult, shoe_type: str = "casual") -> WashPlan:
    """
    shoe_type: casual | sports (two-way; legacy \"leather\" treated as casual).
    Chooses hard vs soft from soil score + material class.
    """
    cfg = load_config()
    if bool(cfg.get("vision", {}).get("rule_based_pipeline", False)) and vision.dirt_level:
        return _decide_wash_rule_based(vision, shoe_type, cfg)
    w = cfg.get("wash", {})
    hard_thr = float(w.get("hard_if_dirt_above", 0.35))
    sport_push = float(w.get("sports_hard_if_dirt_above", 0.22))
    casual_push = float(w.get("casual_hard_if_dirt_above", 0.28))
    st = _normalize_shoe_type(shoe_type)
    d = float(vision.dirt_score)

    if d >= hard_thr:
        return WashPlan(
            "hard",
            f"Heavy soil ({d:.2f}) — deep wash recommended for this {st} shoe",
        )

    if st == "sports":
        if d >= sport_push:
            return WashPlan(
                "hard",
                f"Sports shoe — mesh/textiles with visible soil ({d:.2f}); deeper clean",
            )
        return WashPlan(
            "soft",
            "Sports shoe — light soil; standard gentle wash",
        )
    if st == "casual":
        if d >= casual_push:
            return WashPlan(
                "hard",
                f"Casual shoe — moderate soil ({d:.2f}); stronger wash",
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
    labels = {"casual": "Casual", "sports": "Sports"}
    st = labels.get(_normalize_shoe_type(shoe_type), "Casual")
    if mode == "hard":
        return f"Deep wash ({st})"
    return f"Gentle wash ({st})"
