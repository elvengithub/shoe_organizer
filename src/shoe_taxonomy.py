"""
Map catalog folders + OpenCV cues to display buckets: leather, sports, dress, casual, unknown.
"""
from __future__ import annotations

from .vision_service import ShoeCategory, VisionResult

# API / storage value → short UI title
SHOE_TYPE_LABELS: dict[str, str] = {
    "dress": "Dress",
    "leather": "Leather",
    "sports": "Sports",
    "casual": "Casual",
    "unknown": "Other",
}


def resolve_shoe_type(
    catalog_category: str | None,
    catalog_style: str | None,
    vision: VisionResult,
) -> tuple[str, str]:
    """
    Returns (type_key, short_label). type_key ∈ dress, leather, sports, casual, unknown.
    """
    cc = (catalog_category or "").lower()
    cs = (catalog_style or "").lower()

    if "dress" in cc:
        return "dress", SHOE_TYPE_LABELS["dress"]
    if "casual" in cc:
        if any(k in cs for k in ("sneaker", "trainer", "running", "sport", "athletic")):
            return "sports", SHOE_TYPE_LABELS["sports"]
        return "casual", SHOE_TYPE_LABELS["casual"]
    if "boot" in cc:
        if any(k in cs for k in ("chelsea", "oxford", "dress", "brogue")):
            return "dress", SHOE_TYPE_LABELS["dress"]
        if any(k in cs for k in ("hiking", "trail", "running")):
            return "sports", SHOE_TYPE_LABELS["sports"]
        return "casual", SHOE_TYPE_LABELS["casual"]

    if vision.category == ShoeCategory.LEATHER:
        return "leather", SHOE_TYPE_LABELS["leather"]
    if vision.category == ShoeCategory.SPORTS:
        return "sports", SHOE_TYPE_LABELS["sports"]
    return "unknown", SHOE_TYPE_LABELS["unknown"]


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
