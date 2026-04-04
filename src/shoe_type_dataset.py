"""
Match a live (preprocessed) frame to reference images under datasets/shoe_types/{sports,casual,leather}/.

Uses the same HSV + gray + LAB histogram fusion as shoe_catalog. Add several photos per type taken
from your booth (similar crop/lighting) for best results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_loader import load_config
from .shoe_catalog import _compute_hists, _hist_score, _read_bgr
from .vision_preprocess import apply_vision_preprocess

log = logging.getLogger(__name__)

_ALLOWED_TYPES = frozenset({"sports", "casual", "leather"})

_EXT = {".avif", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_gallery_cache: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] | None = None
_cache_key: tuple[float, float] | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _tree_mtime(root: Path) -> float:
    if not root.is_dir():
        return 0.0
    m = root.stat().st_mtime
    for p in root.rglob("*"):
        if p.is_file():
            try:
                m = max(m, p.stat().st_mtime)
            except OSError:
                pass
    return m


def _config_mtime() -> float:
    p = _project_root() / "config.yaml"
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _hist_cfg(cfg: dict) -> dict:
    block = cfg.get("shoe_type_dataset", {})
    sc = cfg.get("shoe_catalog", {})
    return {
        "resize": int(block.get("resize", sc.get("resize", 160))),
        "hist_h_bins": int(block.get("hist_h_bins", sc.get("hist_h_bins", 24))),
        "hist_s_bins": int(block.get("hist_s_bins", sc.get("hist_s_bins", 24))),
        "hist_gray_bins": int(block.get("hist_gray_bins", sc.get("hist_gray_bins", 32))),
        "hist_lab_bins": int(block.get("hist_lab_bins", sc.get("hist_lab_bins", 16))),
        "weight_hsv": float(block.get("weight_hsv", sc.get("weight_hsv", 0.48))),
        "weight_gray": float(block.get("weight_gray", sc.get("weight_gray", 0.22))),
        "weight_lab": float(block.get("weight_lab", sc.get("weight_lab", 0.30))),
        "blend_bhattacharyya": bool(block.get("blend_bhattacharyya", sc.get("blend_bhattacharyya", False))),
        "bhattacharyya_mix": float(block.get("bhattacharyya_mix", sc.get("bhattacharyya_mix", 0.25))),
    }


def _load_type_galleries(cfg: dict) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    global _gallery_cache, _cache_key
    root = _project_root() / str(cfg.get("shoe_type_dataset", {}).get("path", "datasets/shoe_types"))
    key = (_tree_mtime(root), _config_mtime())
    if _gallery_cache is not None and _cache_key == key:
        return _gallery_cache

    hc = _hist_cfg(cfg)
    out: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {t: [] for t in _ALLOWED_TYPES}
    if not root.is_dir():
        log.warning("shoe_type_dataset path missing: %s", root)
        _gallery_cache = out
        _cache_key = key
        return out

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        tkey = sub.name.lower()
        if tkey not in _ALLOWED_TYPES:
            log.debug("skip unknown shoe_types folder: %s", sub.name)
            continue
        for path in sorted(sub.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _EXT:
                continue
            bgr = _read_bgr(path)
            if bgr is None or bgr.size == 0:
                log.warning("skip unreadable type image: %s", path)
                continue
            bgr = apply_vision_preprocess(bgr, cfg)
            out[tkey].append(_compute_hists(bgr, hc))

    n = sum(len(v) for v in out.values())
    log.info("shoe_type_dataset loaded: %d refs in %s", n, root)
    _gallery_cache = out
    _cache_key = key
    return out


@dataclass
class ShoeTypeDatasetMatch:
    matched: bool
    shoe_type: str | None
    score: float
    scores_by_type: dict[str, float]


def match_shoe_type_from_dataset(
    bgr: np.ndarray,
    cfg: dict | None = None,
    *,
    already_preprocessed: bool = False,
) -> ShoeTypeDatasetMatch:
    cfg = cfg or load_config()
    block = cfg.get("shoe_type_dataset", {})
    if not bool(block.get("enabled", True)):
        return ShoeTypeDatasetMatch(False, None, 0.0, {})

    if not already_preprocessed:
        bgr = apply_vision_preprocess(bgr, cfg)

    galleries = _load_type_galleries(cfg)
    hc = _hist_cfg(cfg)
    q_h, q_g, q_l = _compute_hists(bgr, hc)

    scores_by_type = compute_type_histogram_scores(
        q_h, q_g, q_l, galleries, hc, block
    )

    # Only consider types that have at least one reference
    candidates = {k: v for k, v in scores_by_type.items() if v >= 0}
    if not candidates:
        return ShoeTypeDatasetMatch(False, None, 0.0, scores_by_type)

    sorted_types = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    winner, win_score = sorted_types[0]
    thr = float(block.get("min_match_score", 0.32))
    if win_score < thr:
        return ShoeTypeDatasetMatch(False, None, float(win_score), scores_by_type)

    margin = float(block.get("min_margin", 0.0))
    if margin > 0 and len(sorted_types) > 1:
        second = sorted_types[1][1]
        if win_score - second < margin:
            return ShoeTypeDatasetMatch(False, None, float(win_score), scores_by_type)

    return ShoeTypeDatasetMatch(True, winner, float(win_score), scores_by_type)


def compute_type_histogram_scores(
    q_h: np.ndarray,
    q_g: np.ndarray,
    q_l: np.ndarray,
    galleries: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    hc: dict,
    block: dict,
) -> dict[str, float]:
    """
    Best mean-of-top-k histogram similarity per type (reduces one bad reference dominating).
    Types with no images get score -1.0.
    """
    top_k = max(1, int(block.get("top_k_refs", 2)))
    scores_by_type: dict[str, float] = {}
    for tkey in sorted(_ALLOWED_TYPES):
        refs = galleries.get(tkey) or []
        if not refs:
            scores_by_type[tkey] = -1.0
            continue
        per_ref: list[float] = []
        for rh, rg, rl in refs:
            per_ref.append(_hist_score(q_h, q_g, q_l, rh, rg, rl, hc))
        per_ref.sort(reverse=True)
        k = min(top_k, len(per_ref))
        scores_by_type[tkey] = float(sum(per_ref[:k]) / k)
    return scores_by_type


def compute_type_histogram_scores_from_bgr(
    bgr: np.ndarray,
    cfg: dict,
    *,
    already_preprocessed: bool = False,
) -> dict[str, float]:
    """Raw histogram scores per type (for fusion); -1 if folder empty."""
    if not already_preprocessed:
        bgr = apply_vision_preprocess(bgr, cfg)
    galleries = _load_type_galleries(cfg)
    hc = _hist_cfg(cfg)
    block = cfg.get("shoe_type_dataset", {})
    q_h, q_g, q_l = _compute_hists(bgr, hc)
    return compute_type_histogram_scores(q_h, q_g, q_l, galleries, hc, block)
