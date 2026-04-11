"""
Match a live (preprocessed) frame to reference images under datasets/shoe_types/{sports,casual}/.
Legacy folder "leather/" is merged into casual references.

When ``shoe_type_dataset.reference_subfolder`` (default ``clean``) exists under a type
folder and contains at least one image, only that subfolder is loaded so ``clean/``
booth photos are not mixed with older images in the parent directory.

Uses the same HSV + gray + LAB histogram fusion as shoe_catalog. Add several photos per type
taken from your booth (similar crop/lighting) for best results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_loader import load_config
from .shoe_catalog import _compute_hists, _hist_score, _read_bgr
from .vision_preprocess import apply_vision_preprocess
from .vision_service import leather_like_casual_preferred, leather_like_strong_casual_override

log = logging.getLogger(__name__)

_ALLOWED_TYPES = frozenset({"sports", "casual"})

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


def _iter_type_image_paths(type_dir: Path, block: dict) -> tuple[list[Path], bool]:
    """
    Image paths for one shoe type folder (e.g. casual/ or sports/).

    If ``reference_subfolder`` (default ``clean``) exists under type_dir and has at
    least one image, only that subfolder is used so clean booth-style references
    are not averaged with legacy/dirty images in the parent folder.
    """
    rs = block.get("reference_subfolder", "clean")
    sub_name = "" if rs is None else str(rs).strip().strip("/\\")
    used_clean_only = False
    if not sub_name:
        paths = sorted(type_dir.rglob("*"))
    else:
        clean_dir = type_dir / sub_name
        has_clean = False
        if clean_dir.is_dir():
            for p in clean_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in _EXT:
                    has_clean = True
                    break
        if has_clean:
            paths = sorted(clean_dir.rglob("*"))
            used_clean_only = True
        else:
            paths = sorted(type_dir.rglob("*"))
    out_paths: list[Path] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in _EXT:
            continue
        out_paths.append(path)
    return out_paths, used_clean_only


def _load_type_galleries(cfg: dict) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    global _gallery_cache, _cache_key
    block = cfg.get("shoe_type_dataset", {}) or {}
    root = _project_root() / str(block.get("path", "datasets/shoe_types"))
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
        if tkey == "leather":
            tkey = "casual"
        if tkey not in _ALLOWED_TYPES:
            log.debug("skip unknown shoe_types folder: %s", sub.name)
            continue
        image_paths, using_clean_only = _iter_type_image_paths(sub, block)
        rs = block.get("reference_subfolder", "clean")
        sub_name = "" if rs is None else str(rs).strip().strip("/\\")
        # Do not merge legacy leather/* stock photos into casual when using clean/ refs
        # and leather has no leather/clean/ (would dilute casual/clean histograms).
        if (
            sub.name.lower() == "leather"
            and sub_name
            and not using_clean_only
        ):
            log.debug("skip leather/ (no %s/ refs while reference_subfolder is set)", sub_name)
            continue
        for path in image_paths:
            bgr = _read_bgr(path)
            if bgr is None or bgr.size == 0:
                log.warning("skip unreadable type image: %s", path)
                continue
            try:
                bgr = apply_vision_preprocess(bgr, cfg)
                out[tkey].append(_compute_hists(bgr, hc))
            except Exception as e:
                log.warning("skip shoe_types image (preprocess/hist failed) %s: %s", path, e)
        if using_clean_only:
            log.info("shoe_type_dataset %s: using %s/ only (%d images)", tkey, sub_name, len(out[tkey]))

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
    try:
        q_h, q_g, q_l = _compute_hists(bgr, hc)
        scores_by_type = compute_type_histogram_scores(
            q_h, q_g, q_l, galleries, hc, block
        )
    except Exception as e:
        log.warning("shoe_type_dataset histogram compare failed: %s", e)
        return ShoeTypeDatasetMatch(False, None, 0.0, {})

    # Only consider types that have at least one reference
    candidates = {k: v for k, v in scores_by_type.items() if v >= 0}
    if not candidates:
        return ShoeTypeDatasetMatch(False, None, 0.0, scores_by_type)

    sorted_types = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    winner, win_score = sorted_types[0]
    thr = float(block.get("min_match_score", 0.32))

    forced_leather_casual = False
    if bool(block.get("leather_like_override_dataset_sports", True)) and winner == "sports":
        casual_s = float(scores_by_type.get("casual", -1.0))
        gap = (win_score - casual_s) if casual_s >= 0.0 else 999.0
        soft_gap = float(block.get("leather_like_histogram_gap_soft", 0.22))
        min_casual_flip = float(block.get("leather_like_min_casual_score_flip", 0.14))

        if casual_s >= 0.0 and leather_like_strong_casual_override(bgr, cfg):
            winner, win_score = "casual", casual_s
            forced_leather_casual = True
        elif leather_like_casual_preferred(bgr, cfg) and casual_s >= 0.0:
            if gap < soft_gap or casual_s >= min_casual_flip:
                winner, win_score = "casual", casual_s
                forced_leather_casual = True

    if win_score < thr:
        if not (
            forced_leather_casual
            and winner == "casual"
            and bool(block.get("leather_like_accept_weak_casual_histogram", True))
        ):
            return ShoeTypeDatasetMatch(False, None, float(win_score), scores_by_type)

    margin = float(block.get("min_margin", 0.0))
    if margin > 0 and len(sorted_types) > 1 and not forced_leather_casual:
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
