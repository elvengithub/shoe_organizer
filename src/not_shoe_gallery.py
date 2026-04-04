"""
Phase 1: reference images of non-shoes (bottle, hand, empty props) in datasets/not_shoe/.
If the live ROI matches any negative above min_score_to_reject, we treat as not a shoe.
Uses the same histogram settings as shoe_catalog.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config_loader import load_config
from .shoe_catalog import _compute_hists, _hist_score, _read_bgr
from .vision_preprocess import apply_vision_preprocess

log = logging.getLogger(__name__)

_EXT = {".avif", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_neg_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
_neg_cache_key: tuple[float, float] | None = None


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


def _neg_key(cfg: dict) -> tuple[float, float]:
    ns = cfg.get("not_shoe_catalog", {})
    rel = ns.get("path", "datasets/not_shoe")
    root = _project_root() / rel
    return (_tree_mtime(root), _config_mtime())


def _load_negative_hists(cfg: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    global _neg_cache, _neg_cache_key
    key = _neg_key(cfg)
    if _neg_cache is not None and _neg_cache_key == key:
        return _neg_cache

    ns = cfg.get("not_shoe_catalog", {})
    rel = ns.get("path", "datasets/not_shoe")
    root = _project_root() / rel
    sc = cfg.get("shoe_catalog", {})

    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _EXT:
                continue
            bgr = _read_bgr(path)
            if bgr is None or bgr.size == 0:
                continue
            bgr = apply_vision_preprocess(bgr, cfg)
            h, g, l = _compute_hists(bgr, sc)
            out.append((h, g, l))

    _neg_cache = out
    _neg_cache_key = key
    log.info("not-shoe gallery: %d reference images from %s", len(out), root)
    return out


def max_not_shoe_similarity(bgr_preprocessed: np.ndarray, cfg: dict | None = None) -> float:
    cfg = cfg or load_config()
    ns = cfg.get("not_shoe_catalog", {})
    if not bool(ns.get("enabled", True)):
        return 0.0

    entries = _load_negative_hists(cfg)
    if not entries:
        return 0.0

    sc = cfg.get("shoe_catalog", {})
    q_h, q_g, q_l = _compute_hists(bgr_preprocessed, sc)
    best = -1.0
    for e_h, e_g, e_l in entries:
        s = _hist_score(q_h, q_g, q_l, e_h, e_g, e_l, sc)
        best = max(best, s)
    return float(best)
