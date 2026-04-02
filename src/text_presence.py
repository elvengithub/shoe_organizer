"""
Classify what is presented using only a free-text description (no camera images).
Rule: only clear shoe wording counts as a shoe; anything else → Not a shoe.
"""
from __future__ import annotations

import re
from typing import Any

from .ai_camera import NOT_SHOE_MESSAGE
from .config_loader import load_config
from .shoe_taxonomy import SHOE_TYPE_LABELS, format_shoe_display_name
from .vision_service import ShoeCategory, VisionResult
from .wash_decision import decide_wash, wash_ui_label

_DEFAULT_NOT_SHOE = frozenset(
    """
    bottle bag backpack box phone cup mug plate fork food fruit apple banana
    hand hands face head person human body arm leg cat dog animal toy ball
    rock stone brick tool hammer screwdriver keys wallet hat cap glove empty
    nothing air vase book laptop tablet remote watch belt sock socks alone
    water coffee paper trash can bin umbrella jacket shirt pants jeans
    glasses sunglasses car key chain charger cable wood metal plastic
    """.split()
)

_DEFAULT_SHOE = frozenset(
    """
    shoe shoes sneaker sneakers boot boots loafer loafers oxford oxfords
    brogue brogues derby derbies sandal sandals heel heels pump pumps
    stiletto cleat cleats slipper slippers mule mules flat flats athletic
    running trainer trainers tennis basketball skate skateboarding chukka
    wingtip monkstrap espadrille clog clogs crocs jordan airforce yeezy
    """.split()
)

_DRESS_HINTS = frozenset("dress oxford brogue formal pump stiletto heel wingtip monk loafer patent evening".split())
_SPORTS_HINTS = frozenset("sneaker athletic running trainer tennis basketball cleat skate gym sport jog".split())
_LEATHER_HINTS = frozenset("leather patent suede cowhide".split())
_CASUAL_HINTS = frozenset("casual sandal flip flop slipper flat mule clog croc boot ankle hiking trail".split())

_DIRTY_HINTS = frozenset("dirty mud muddy soiled stained grimy scuffed heavy worn filthy".split())
_CLEAN_HINTS = frozenset("clean new pristine fresh barely".split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hits(tokens: list[str], vocab: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in vocab)


def _infer_dirt_score(tokens: list[str], cfg: dict) -> float:
    base = float(cfg.get("text_classifier", {}).get("default_dirt_score", 0.18))
    if any(w in tokens for w in _DIRTY_HINTS) or any(
        any(d in t for d in ("mud", "dirt", "soil", "stain")) for t in tokens
    ):
        return min(0.85, base + 0.45)
    if any(w in tokens for w in _CLEAN_HINTS):
        return max(0.05, base - 0.1)
    return base


def _infer_shoe_type(tokens: list[str]) -> str:
    ts = " ".join(tokens)
    if any(h in ts for h in _DRESS_HINTS):
        return "dress"
    if any(h in ts for h in _SPORTS_HINTS):
        return "sports"
    if any(h in ts for h in _LEATHER_HINTS):
        return "leather"
    if any(h in ts for h in _CASUAL_HINTS):
        return "casual"
    if "shoe" in tokens or "shoes" in tokens or "boot" in tokens or "boots" in tokens:
        return "casual"
    return "unknown"


def _style_snippet(raw: str, max_len: int = 48) -> str:
    t = " ".join(raw.split())
    return (t[:max_len] + "…") if len(t) > max_len else t


def analyze_presented_text(description: str, cfg: dict | None = None) -> dict[str, Any]:
    """
    Returns API-shaped dict (same keys as camera analyze success path).
    Unknown / non-shoe wording → not_shoe. No images used.
    """
    cfg = cfg or load_config()
    tc = cfg.get("text_classifier", {})
    raw = (description or "").strip()
    if not raw:
        return {
            "ok": False,
            "error": "empty_description",
            "message": "Describe what is presented in the bay (text only).",
            "input_mode": "text",
        }

    not_vocab = frozenset(tc.get("not_shoe_keywords", [])) | _DEFAULT_NOT_SHOE
    shoe_vocab = frozenset(tc.get("shoe_keywords", [])) | _DEFAULT_SHOE

    tokens = _tokens(raw)
    if not tokens:
        return {
            "ok": True,
            "error": "not_shoe",
            "is_shoe": False,
            "raw_is_shoe": False,
            "message": NOT_SHOE_MESSAGE,
            "object_classification": "not_shoe",
            "input_mode": "text",
            "reject_stage": "text_no_tokens",
        }

    n_hit = _hits(tokens, not_vocab)
    s_hit = _hits(tokens, shoe_vocab)

    # Require strictly more shoe signal than non-shoe; ties → not a shoe.
    if s_hit == 0 or n_hit >= s_hit:
        return {
            "ok": True,
            "error": "not_shoe",
            "is_shoe": False,
            "raw_is_shoe": False,
            "message": NOT_SHOE_MESSAGE,
            "object_classification": "not_shoe",
            "input_mode": "text",
            "reject_stage": "text_not_shoe_keywords",
        }

    shoe_type = _infer_shoe_type(tokens)
    d = _infer_dirt_score(tokens, cfg)
    vision = VisionResult(dirt_score=d, category=ShoeCategory.UNKNOWN)
    wash = decide_wash(vision, shoe_type)
    sl = SHOE_TYPE_LABELS.get(shoe_type, shoe_type.title())
    style = _style_snippet(raw)
    shoe_type_label = format_shoe_display_name(shoe_type, sl, "Description", style)

    detail: dict[str, Any] = {
        "ok": True,
        "error": None,
        "is_shoe": True,
        "raw_is_shoe": True,
        "reject_stage": None,
        "catalog_match": True,
        "object_classification": f"shoe_{shoe_type}",
        "shoe_category": shoe_type,
        "shoe_type_label": shoe_type_label,
        "shoe_type_short": sl,
        "dirt_score": round(d, 4),
        "wash_mode": wash.mode,
        "wash_label": wash_ui_label(wash.mode, shoe_type),
        "wash_reason": wash.reason,
        "catalog_category": "Description",
        "catalog_style": style,
        "catalog_score": 1.0,
        "inference_backend": "text_keywords",
        "gate_reason": "text_only",
        "confirmed_is_shoe": True,
        "stabilizing": False,
        "input_mode": "text",
    }
    return detail
