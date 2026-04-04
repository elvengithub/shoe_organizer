"""
Map catalog folders + OpenCV cues to exactly three buckets: casual, sports, leather.
"""
from __future__ import annotations

from .vision_service import ShoeCategory, VisionResult

# API / storage value → short UI title (three types only)
SHOE_TYPE_LABELS: dict[str, str] = {
    "casual": "Casual",
    "sports": "Sports",
    "leather": "Leather",
}


def resolve_shoe_type(
    catalog_category: str | None,
    catalog_style: str | None,
    vision: VisionResult,
) -> tuple[str, str]:
    """
    Returns (type_key, short_label). type_key is always casual | sports | leather.
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
    if "dress" in cc or "formal" in cc:
        return "leather", SHOE_TYPE_LABELS["leather"]
    if "casual" in cc:
        return ("sports", SHOE_TYPE_LABELS["sports"]) if athletic_style() else ("casual", SHOE_TYPE_LABELS["casual"])
    if "leather" in cc or "boot" in cc:
        if athletic_style() or any(k in cs for k in ("hiking", "trail", "running")):
            return "sports", SHOE_TYPE_LABELS["sports"]
        if any(k in cs for k in ("chelsea", "oxford", "dress", "brogue", "wingtip", "monk", "loafer")):
            return "leather", SHOE_TYPE_LABELS["leather"]
        return "casual", SHOE_TYPE_LABELS["casual"]

    # Vision fallback when catalog is weak or style string empty
    if vision.category == ShoeCategory.LEATHER:
        return "leather", SHOE_TYPE_LABELS["leather"]
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
