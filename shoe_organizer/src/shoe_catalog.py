"""
Match live frames to datasets/shoes/<category>/<name>.* using HSV + gray + LAB histograms
after optional ROI + CLAHE (vision_preprocess). Tune shoe_catalog + vision_preprocess in config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config_loader import load_config
from .vision_preprocess import apply_vision_preprocess

log = logging.getLogger(__name__)

_EXT = {".avif", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_gallery_cache: list["_GalleryEntry"] | None = None
_gallery_cache_key: tuple[float, float] | None = None


@dataclass
class CatalogMatch:
    matched: bool
    category: str | None
    style: str | None
    score: float


@dataclass
class _GalleryEntry:
    category: str
    style: str
    h_hist: np.ndarray
    g_hist: np.ndarray
    lab_hist: np.ndarray


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _gallery_tree_mtime(root: Path) -> float:
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


def _cache_key(cfg: dict) -> tuple[float, float]:
    sc = cfg.get("shoe_catalog", {})
    rel = sc.get("path", "datasets/shoes")
    root = _project_root() / rel
    return (_gallery_tree_mtime(root), _config_mtime())


def _read_bgr(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.debug("PIL load failed %s: %s", path, e)
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _compute_hists(bgr: np.ndarray, sc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rsz = max(32, int(sc.get("resize", 160)))
    small = cv2.resize(bgr, (rsz, rsz), interpolation=cv2.INTER_AREA)
    hb = int(sc.get("hist_h_bins", 24))
    sb = int(sc.get("hist_s_bins", 24))
    gb = int(sc.get("hist_gray_bins", 32))
    lb = int(sc.get("hist_lab_bins", 16))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0, 1], None, [hb, sb], [0, 180, 0, 256])
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g_hist = cv2.calcHist([gray], [0], None, [gb], [0, 256])
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lab_hist = cv2.calcHist([lab], [1, 2], None, [lb, lb], [0, 256, 0, 256])

    cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(lab_hist, lab_hist, 0, 1, cv2.NORM_MINMAX)
    return h_hist, g_hist, lab_hist


def _correl(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL)))


def _hist_score(
    q_h: np.ndarray,
    q_g: np.ndarray,
    q_l: np.ndarray,
    e_h: np.ndarray,
    e_g: np.ndarray,
    e_l: np.ndarray,
    sc: dict,
) -> float:
    wh = float(sc.get("weight_hsv", 0.48))
    wg = float(sc.get("weight_gray", 0.22))
    wl = float(sc.get("weight_lab", 0.30))
    s = wh + wg + wl
    if s <= 0:
        s = 1.0
    wh, wg, wl = wh / s, wg / s, wl / s
    c_h = _correl(q_h, e_h)
    c_g = _correl(q_g, e_g)
    c_l = _correl(q_l, e_l)
    score = wh * c_h + wg * c_g + wl * c_l
    if bool(sc.get("blend_bhattacharyya", False)):
        b_h = max(0.0, 1.0 - float(cv2.compareHist(q_h, e_h, cv2.HISTCMP_BHATTACHARYYA)))
        b_g = max(0.0, 1.0 - float(cv2.compareHist(q_g, e_g, cv2.HISTCMP_BHATTACHARYYA)))
        b_l = max(0.0, 1.0 - float(cv2.compareHist(q_l, e_l, cv2.HISTCMP_BHATTACHARYYA)))
        mix = float(sc.get("bhattacharyya_mix", 0.25))
        score = (1.0 - mix) * score + mix * (wh * b_h + wg * b_g + wl * b_l)
    return score


def _load_gallery(cfg: dict) -> list[_GalleryEntry]:
    global _gallery_cache, _gallery_cache_key
    key = _cache_key(cfg)
    if _gallery_cache is not None and _gallery_cache_key == key:
        return _gallery_cache

    sc = cfg.get("shoe_catalog", {})
    rel = sc.get("path", "datasets/shoes")
    root = _project_root() / rel
    if not root.is_dir():
        log.warning("shoe catalog path missing: %s", root)
        _gallery_cache = []
        _gallery_cache_key = key
        return []

    entries: list[_GalleryEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _EXT:
            continue
        try:
            rel_p = path.relative_to(root)
        except ValueError:
            continue
        parts = rel_p.parts
        if len(parts) >= 2:
            category = parts[0]
            style = Path(parts[-1]).stem
        else:
            category = "General"
            style = path.stem
        bgr = _read_bgr(path)
        if bgr is None or bgr.size == 0:
            log.warning("skip unreadable catalog image: %s", path)
            continue
        try:
            bgr = apply_vision_preprocess(bgr, cfg)
            h_hist, g_hist, lab_hist = _compute_hists(bgr, sc)
            entries.append(
                _GalleryEntry(category=category, style=style, h_hist=h_hist, g_hist=g_hist, lab_hist=lab_hist)
            )
        except Exception as e:
            log.warning("skip catalog image (preprocess/hist failed) %s: %s", path, e)

    _gallery_cache = entries
    _gallery_cache_key = key
    log.info("shoe catalog loaded: %d reference images from %s", len(entries), root)
    return entries


def match_against_catalog(
    bgr: np.ndarray,
    cfg: dict | None = None,
    *,
    already_preprocessed: bool = False,
) -> CatalogMatch:
    cfg = cfg or load_config()
    sc = cfg.get("shoe_catalog", {})
    if not bool(sc.get("enabled", True)):
        return CatalogMatch(True, None, None, 1.0)

    entries = _load_gallery(cfg)
    if not entries:
        log.warning("shoe catalog empty — cannot verify a shoe match")
        return CatalogMatch(False, None, None, 0.0)

    if not already_preprocessed:
        bgr = apply_vision_preprocess(bgr, cfg)
    q_h, q_g, q_l = _compute_hists(bgr, sc)
    scored: list[tuple[float, _GalleryEntry]] = []
    for e in entries:
        s = _hist_score(q_h, q_g, q_l, e.h_hist, e.g_hist, e.lab_hist, sc)
        scored.append((s, e))
    scored.sort(key=lambda t: t[0], reverse=True)

    best_score, ent = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0

    thr = float(sc.get("min_match_score", 0.38))
    if ent is None or best_score < thr:
        return CatalogMatch(False, None, None, float(best_score))

    weak_cap = float(sc.get("reject_if_ambiguous_below", 0.52))
    min_margin = float(sc.get("min_score_margin_when_weak", 0.035))
    if len(scored) >= 2 and second_score >= 0.0 and best_score < weak_cap:
        if (best_score - second_score) < min_margin:
            return CatalogMatch(False, None, None, float(best_score))

    return CatalogMatch(True, ent.category, ent.style, float(best_score))
