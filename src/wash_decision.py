from __future__ import annotations

from dataclasses import dataclass

from .config_loader import load_config
from .vision_service import VisionResult


@dataclass
class WashPlan:
    mode: str
    reason: str


def decide_wash(vision: VisionResult, shoe_type: str = "unknown") -> WashPlan:
    """
    shoe_type: dress | leather | sports | casual | unknown
    Chooses hard vs soft from soil score + material class.
    """
    cfg = load_config()
    w = cfg.get("wash", {})
    hard_thr = float(w.get("hard_if_dirt_above", 0.35))
    sport_push = float(w.get("sports_hard_if_dirt_above", 0.22))
    casual_push = float(w.get("casual_hard_if_dirt_above", 0.28))
    st = (shoe_type or "unknown").lower()
    d = float(vision.dirt_score)

    if d >= hard_thr:
        return WashPlan(
            "hard",
            f"Heavy soil ({d:.2f}) — deep wash recommended for this {st} shoe",
        )

    if st == "dress":
        return WashPlan(
            "soft",
            "Dress shoe — gentle wash to protect finish and structure",
        )
    if st == "leather":
        return WashPlan(
            "soft",
            "Leather — gentle cycle to reduce scuffing and drying",
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
        f"General care — gentle wash (soil {d:.2f})",
    )


def wash_ui_label(mode: str, shoe_type: str) -> str:
    st = (shoe_type or "unknown").title()
    if mode == "hard":
        return f"Deep wash ({st})"
    return f"Gentle wash ({st})"
