"""
Map catalog folders + OpenCV cues to two buckets: casual, sports.
"""
from __future__ import annotations

from .vision_service import ShoeCategory, VisionResult

# API / storage value → short UI title (two types only)
SHOE_TYPE_LABELS: dict[str, str] = {
    "casual": "Casual",
    "sports": "Sports",
}


def resolve_shoe_type(
    catalog_category: str | None,
    catalog_style: str | None,
    vision: VisionResult,
) -> tuple[str, str]:
    """
    Returns (type_key, short_label). type_key is always casual | sports.
    """
    cc = (catalog_category or "").lower()
    cs = (catalog_style or "").lower()

    def athletic_style() -> bool:
        return any(
            k in cs
            for k in (
                "sneaker",
                "trainer",
                "running",
                "runner",
                "runners",
                "sport",
                "athletic",
                "basketball",
                "tennis",
                "cleat",
                "skate",
                "gym",
                "jog",
                "hiking",
                "trail",
            )
        )

    # Catalog path hints (folder names from datasets/shoes/...)
    if "casual" in cc:
        return ("sports", SHOE_TYPE_LABELS["sports"]) if athletic_style() else ("casual", SHOE_TYPE_LABELS["casual"])
    if "leather" in cc or "dress" in cc or "formal" in cc or "boot" in cc:
        if athletic_style() or any(k in cs for k in ("hiking", "trail", "running")):
            return "sports", SHOE_TYPE_LABELS["sports"]
        return "casual", SHOE_TYPE_LABELS["casual"]

    # Vision fallback when catalog is weak or style string empty
    if vision.category == ShoeCategory.SPORTS:
        return "sports", SHOE_TYPE_LABELS["sports"]
    return "casual", SHOE_TYPE_LABELS["casual"]


def format_shoe_display_name(
    type_key: str,
    type_label: str,
    catalog_category: str | None,
    catalog_style: str | None,
) -> str:
    if catalog_category and catalog_style:
        return f"{type_label} — {catalog_style}"
    if catalog_style:
        return f"{type_label} — {catalog_style}"
    return type_label
