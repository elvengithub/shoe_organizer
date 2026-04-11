from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .config_loader import load_config
from .shoe_onnx_gate import try_onnx_shoe_gate

log = logging.getLogger(__name__)


class ShoeCategory(str, Enum):
    """Camera-only bucket: sports | casual (two-way)."""

    SPORTS = "sports"
    CASUAL = "casual"


@dataclass
class VisionResult:
    dirt_score: float
    category: ShoeCategory
    frame_bgr: np.ndarray | None = None
    # Rule-based pipeline (OpenCV only)
    dirt_level: str | None = None  # clean | moderate | very_dirty
    edge_density: float | None = None
    dirty_pixel_ratio: float | None = None
    # Multi-cue sports vs casual (when rule_based_pipeline)
    sports_fusion_score: float | None = None
    sports_fusion_threshold: float | None = None
    gradient_mean: float | None = None
    texture_rms: float | None = None
    saturation_mean: float | None = None
    # 1.0 when frame matched leather/dress-like muted cues (maps to casual, not sports)
    leather_like_casual: float | None = None


@dataclass
class ShoeGateResult:
    """Contour gate and/or ONNX shoe detector — tune `shoe_gate` and optional `shoe_object_detection`."""

    is_shoe: bool
    reason: str


def evaluate_shoe_gate(bgr: np.ndarray, cfg: dict | None = None) -> ShoeGateResult:
    cfg = cfg or load_config()
    od = try_onnx_shoe_gate(bgr, cfg)
    if od is not None:
        return od

    sg = cfg.get("shoe_gate", {})
    if not bool(sg.get("enabled", True)):
        return ShoeGateResult(True, "gate_disabled")

    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return ShoeGateResult(False, "frame_too_small")

    area_frame = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ShoeGateResult(False, "no_object")

    cnt = max(contours, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt))
    ratio = a / area_frame
    min_r = float(sg.get("min_contour_area_ratio", 0.025))
    max_r = float(sg.get("max_contour_area_ratio", 0.92))
    if ratio < min_r:
        return ShoeGateResult(False, "object_too_small")
    if ratio > max_r:
        return ShoeGateResult(False, "object_fills_frame")

    _x, _y, rw, rh = cv2.boundingRect(cnt)
    rw = max(rw, 1)
    rh = max(rh, 1)
    ar = max(rw, rh) / min(rw, rh)
    min_ar = float(sg.get("min_aspect_ratio", 1.1))
    max_ar = float(sg.get("max_aspect_ratio", 7.5))
    if ar < min_ar:
        return ShoeGateResult(False, "too_compact")
    if ar > max_ar:
        return ShoeGateResult(False, "too_elongated")

    hull = cv2.convexHull(cnt)
    hull_a = float(cv2.contourArea(hull))
    solidity = a / max(hull_a, 1e-6)
    min_sol = float(sg.get("min_solidity", 0.32))
    if solidity < min_sol:
        return ShoeGateResult(False, "too_irregular")

    return ShoeGateResult(True, "ok")


def _dirt_score(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edge_density = float(np.mean(edges > 0))
    std = float(np.std(gray)) / 255.0
    return min(1.0, 0.5 * edge_density + 0.5 * std)


def compute_edge_density(bgr: np.ndarray) -> float:
    """Fraction of edge pixels (Canny) — used for sports vs smooth shoe heuristics."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return float(np.mean(edges > 0))


def _rule_dirt_level_and_score(bgr: np.ndarray, cfg: dict) -> tuple[float, str, float]:
    """
    Dirt from brown/dark-ish pixels + grayscale variance (no ML).
    Returns (dirt_score 0..1, dirt_level, dirty_pixel_ratio).
    """
    vm = cfg.get("vision", {})
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    # Brown / mud / dirt hues (OpenCV H 0–179)
    h_lo = int(vm.get("dirt_hue_min", 8))
    h_hi = int(vm.get("dirt_hue_max", 32))
    s_min = int(vm.get("dirt_sat_min", 20))
    v_brown_lo = int(vm.get("dirt_value_brown_min", 35))
    v_brown_hi = int(vm.get("dirt_value_brown_max", 220))
    brown = (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_brown_lo) & (v <= v_brown_hi)

    v_dark_max = int(vm.get("dirt_dark_v_max", 75))
    s_any = int(vm.get("dirt_dark_sat_min", 15))
    dark = (v <= v_dark_max) & (s >= s_any)

    dirty_mask = brown | dark
    ratio = float(np.mean(dirty_mask))
    std_norm = float(np.std(gray)) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise = float(np.std(lap)) / 64.0
    w_r = float(vm.get("dirt_blend_ratio", 0.55))
    w_std = float(vm.get("dirt_blend_std", 0.30))
    w_n = float(vm.get("dirt_blend_noise", 0.15))
    combined = w_r * ratio + w_std * std_norm + w_n * min(1.0, noise)

    thr_clean = float(vm.get("dirt_ratio_clean_below", 0.05))
    thr_mod = float(vm.get("dirt_ratio_moderate_below", 0.15))
    if combined < thr_clean:
        level = "clean"
        score = 0.08 + 0.12 * (combined / max(thr_clean, 1e-6))
    elif combined < thr_mod:
        level = "moderate"
        t = (combined - thr_clean) / max(thr_mod - thr_clean, 1e-6)
        score = 0.20 + 0.18 * t
    else:
        level = "very_dirty"
        score = min(1.0, 0.42 + 0.58 * min(1.0, (combined - thr_mod) / max(1.0 - thr_mod, 1e-6)))

    return float(score), level, ratio


def _segment_shoe_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Fast foreground segmentation using adaptive thresholding + edge-aware morphology.
    Returns a binary mask (255 = shoe, 0 = background) same size as bgr.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(blur, 30, 100)
    dilated_edges = cv2.dilate(edges, None, iterations=2)
    combined = cv2.bitwise_or(otsu, dilated_edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=4)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=np.uint8) * 255

    area_frame = float(h * w)
    big_cnts = [c for c in contours if cv2.contourArea(c) > area_frame * 0.03]
    if not big_cnts:
        big_cnts = [max(contours, key=cv2.contourArea)]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, big_cnts, -1, 255, cv2.FILLED)
    return mask


def _extract_shoe_shape_features(mask: np.ndarray) -> dict[str, float]:
    """
    Structural shape features from the shoe silhouette.
    Sports: thick chunky soles, padded collar bulge at top-rear, wide/tall profile,
            high defect count (cutouts, overlays), higher bounding-box fill.
    Casual: thin flat soles, slim uniform profile, low collar, simpler outline.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defaults = {"aspect_ratio": 1.5, "solidity": 0.7, "complexity": 12.0,
                "sole_ratio": 0.2, "contour_area_ratio": 0.3, "sole_thickness_norm": 0.15,
                "collar_bulge": 0.0, "width_variation": 0.1, "convexity_defects": 3.0}
    if not contours:
        return defaults

    cnt = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(cnt)), 1.0)
    peri = max(float(cv2.arcLength(cnt, True)), 1.0)
    x, y, bw, bh = cv2.boundingRect(cnt)
    bw = max(bw, 1)
    bh = max(bh, 1)

    hull = cv2.convexHull(cnt)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)

    h_img, w_img = mask.shape[:2]
    aspect_ratio = float(bw) / float(bh)
    solidity = area / hull_area
    complexity = (peri * peri) / area
    contour_area_ratio = area / float(h_img * w_img)

    # Sole thickness: scan bottom 30% row-by-row for how many rows have shoe pixels
    sole_region = mask[y + int(0.70 * bh):y + bh, x:x + bw]
    if sole_region.size > 0:
        row_has_pixels = np.any(sole_region > 0, axis=1)
        sole_thickness_px = float(np.sum(row_has_pixels))
        sole_thickness_norm = sole_thickness_px / max(float(bh), 1.0)
    else:
        sole_thickness_norm = 0.15

    bottom_quarter_y = y + int(0.75 * bh)
    sole_strip = mask[bottom_quarter_y:y + bh, x:x + bw]
    sole_pixels = float(np.sum(sole_strip > 0))
    shoe_pixels_total = max(float(np.sum(mask[y:y + bh, x:x + bw] > 0)), 1.0)
    sole_ratio = sole_pixels / shoe_pixels_total

    # Collar bulge: width of top 20% vs middle 50%
    top_strip = mask[y:y + int(0.20 * bh), x:x + bw]
    mid_strip = mask[y + int(0.25 * bh):y + int(0.75 * bh), x:x + bw]
    top_width = float(np.mean(np.sum(top_strip > 0, axis=1))) if top_strip.size > 0 else 0
    mid_width = float(np.mean(np.sum(mid_strip > 0, axis=1))) if mid_strip.size > 0 else 1
    collar_bulge = (top_width / max(mid_width, 1.0)) - 1.0

    # Width variation along height (sports have more varied widths due to soles/padding)
    row_widths = []
    for row_y in range(y, y + bh):
        if row_y < h_img:
            row = mask[row_y, x:x + bw]
            row_widths.append(float(np.sum(row > 0)))
    width_variation = float(np.std(row_widths)) / max(float(np.mean(row_widths)), 1.0) if row_widths else 0.1

    # Convexity defects: more defects = more complex shape (overlays, cutouts)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cnt, hull_indices)
        n_defects = len(defects) if defects is not None else 0
    except cv2.error:
        n_defects = 0

    return {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "complexity": complexity,
        "sole_ratio": sole_ratio,
        "contour_area_ratio": contour_area_ratio,
        "sole_thickness_norm": sole_thickness_norm,
        "collar_bulge": collar_bulge,
        "width_variation": width_variation,
        "convexity_defects": float(n_defects),
    }


def _compute_lbp_texture(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Local Binary Pattern (simplified 8-neighbor) texture descriptor on the shoe region.
    Computed on a denoised version to reduce dirt/stain noise and focus on material texture.
    """
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = denoised.shape
    padded = cv2.copyMakeBorder(denoised, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    center = padded[1:h + 1, 1:w + 1].astype(np.int16)

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    lbp = np.zeros((h, w), dtype=np.uint8)
    for i, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx].astype(np.int16)
        lbp += ((neighbor >= center).astype(np.uint8)) << i

    shoe_mask = mask > 0
    if np.sum(shoe_mask) < 100:
        return {"lbp_mean": 0.5, "lbp_std": 0.15, "lbp_uniformity": 0.5}

    lbp_vals = lbp[shoe_mask].astype(np.float32)
    lbp_mean = float(np.mean(lbp_vals)) / 255.0
    lbp_std = float(np.std(lbp_vals)) / 255.0

    hist, _ = np.histogram(lbp_vals, bins=32, range=(0, 256))
    hist_norm = hist.astype(np.float32) / max(float(np.sum(hist)), 1.0)
    uniformity = float(np.sum(hist_norm ** 2))

    return {"lbp_mean": lbp_mean, "lbp_std": lbp_std, "lbp_uniformity": uniformity}


def _compute_color_features_masked(bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Color features computed only on shoe pixels (masked).
    Sports: higher saturation, more color variety, brighter.
    Casual: lower saturation, more uniform, often muted tones.
    """
    shoe_mask = mask > 0
    if np.sum(shoe_mask) < 100:
        return {"sat_mean": 0.3, "sat_std": 0.1, "val_mean": 0.5,
                "hue_std": 0.1, "color_variety": 3.0, "bright_ratio": 0.3}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, :, 0][shoe_mask].astype(np.float32)
    s_vals = hsv[:, :, 1][shoe_mask].astype(np.float32)
    v_vals = hsv[:, :, 2][shoe_mask].astype(np.float32)

    sat_mean = float(np.mean(s_vals)) / 255.0
    sat_std = float(np.std(s_vals)) / 255.0
    val_mean = float(np.mean(v_vals)) / 255.0
    hue_std = float(np.std(h_vals)) / 90.0

    h_bins = np.histogram(h_vals, bins=18, range=(0, 180))[0]
    h_bins_norm = h_bins.astype(np.float32) / max(float(np.sum(h_bins)), 1.0)
    color_variety = float(np.sum(h_bins_norm > 0.02))

    bright_ratio = float(np.sum(v_vals > 180)) / max(float(len(v_vals)), 1.0)

    return {
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "val_mean": val_mean,
        "hue_std": hue_std,
        "color_variety": color_variety,
        "bright_ratio": bright_ratio,
    }


def _compute_edge_texture_masked(bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Edge/gradient/texture features computed only on the shoe region.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    shoe_mask = mask > 0
    if np.sum(shoe_mask) < 100:
        return {"edge_density": 0.1, "gradient_mean": 0.08, "texture_rms": 0.04,
                "high_freq_energy": 0.05}

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_pixels = edges[shoe_mask]
    edge_density = float(np.mean(edge_pixels > 0))

    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    gradient_mean = float(np.mean(grad_mag[shoe_mask])) / 255.0

    bl5 = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - bl5.astype(np.float32)
    texture_rms = float(np.sqrt(np.mean(residual[shoe_mask] ** 2))) / 255.0

    bl3 = cv2.GaussianBlur(gray, (3, 3), 0)
    hf_residual = gray.astype(np.float32) - bl3.astype(np.float32)
    high_freq_energy = float(np.mean(np.abs(hf_residual[shoe_mask]))) / 255.0

    return {
        "edge_density": edge_density,
        "gradient_mean": gradient_mean,
        "texture_rms": texture_rms,
        "high_freq_energy": high_freq_energy,
    }


def _compute_sole_contrast(bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """
    Detect contrast between upper and sole regions — sports shoes typically have
    thick, distinctly colored soles (often white/bright) vs the upper.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"sole_brightness_diff": 0.0, "sole_color_diff": 0.0}

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bh < 10 or bw < 10:
        return {"sole_brightness_diff": 0.0, "sole_color_diff": 0.0}

    upper_y1 = y + int(0.15 * bh)
    upper_y2 = y + int(0.55 * bh)
    sole_y1 = y + int(0.78 * bh)
    sole_y2 = y + bh

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    upper_mask = mask[upper_y1:upper_y2, x:x + bw] > 0
    sole_mask = mask[sole_y1:sole_y2, x:x + bw] > 0

    if np.sum(upper_mask) < 50 or np.sum(sole_mask) < 50:
        return {"sole_brightness_diff": 0.0, "sole_color_diff": 0.0}

    upper_v = float(np.mean(gray[upper_y1:upper_y2, x:x + bw][upper_mask]))
    sole_v = float(np.mean(gray[sole_y1:sole_y2, x:x + bw][sole_mask]))
    sole_brightness_diff = abs(sole_v - upper_v) / 255.0

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    upper_lab = lab[upper_y1:upper_y2, x:x + bw][upper_mask].astype(np.float32)
    sole_lab = lab[sole_y1:sole_y2, x:x + bw][sole_mask].astype(np.float32)
    upper_mean = np.mean(upper_lab, axis=0)
    sole_mean = np.mean(sole_lab, axis=0)
    sole_color_diff = float(np.linalg.norm(sole_mean - upper_mean)) / 255.0

    return {
        "sole_brightness_diff": sole_brightness_diff,
        "sole_color_diff": sole_color_diff,
    }


def _analyze_horizontal_profile(bgr: np.ndarray) -> dict[str, float]:
    """
    Divide the image into horizontal strips and analyze the gradient/edge profile.
    Sports shoes show a strong horizontal edge band at the sole-upper junction and
    high gradient in the bottom strip (treaded outsole). Casual shoes have a
    more uniform gradient distribution top-to-bottom.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    n_strips = 5
    strip_h = h // n_strips
    strip_edge_densities = []
    strip_gradients = []

    for i in range(n_strips):
        y1 = i * strip_h
        y2 = (i + 1) * strip_h if i < n_strips - 1 else h
        strip = blur[y1:y2, :]
        edges = cv2.Canny(strip, 50, 150)
        strip_edge_densities.append(float(np.mean(edges > 0)))
        gy = cv2.Sobel(strip, cv2.CV_64F, 0, 1, ksize=3)
        gx = cv2.Sobel(strip, cv2.CV_64F, 1, 0, ksize=3)
        strip_gradients.append(float(np.mean(np.sqrt(gx * gx + gy * gy))) / 255.0)

    # Horizontal edge strength in sole zone (strip 3 & 4 = bottom 40%)
    bottom_edge = max(strip_edge_densities[3], strip_edge_densities[4]) if n_strips >= 5 else 0
    top_edge = max(strip_edge_densities[0], strip_edge_densities[1]) if n_strips >= 5 else 0
    sole_edge_prominence = bottom_edge - top_edge

    # Gradient concentration in sole area
    bottom_grad = max(strip_gradients[3], strip_gradients[4]) if n_strips >= 5 else 0
    mid_grad = strip_gradients[2] if n_strips >= 5 else 0
    sole_grad_boost = bottom_grad - mid_grad

    # Horizontal line detection (sole line = strong horizontal edge)
    h_kernel = np.ones((1, max(w // 4, 10)), dtype=np.float32) / max(w // 4, 10)
    h_response = cv2.filter2D(blur, cv2.CV_64F, h_kernel)
    h_edges = cv2.Canny(h_response.astype(np.uint8), 30, 100)
    # Focus on bottom 40% where soles are
    h_sole_zone = h_edges[int(0.6 * h):, :]
    sole_line_density = float(np.mean(h_sole_zone > 0)) if h_sole_zone.size > 0 else 0

    return {
        "sole_edge_prominence": sole_edge_prominence,
        "sole_grad_boost": sole_grad_boost,
        "sole_line_density": sole_line_density,
        "bottom_edge_density": bottom_edge,
        "top_edge_density": top_edge,
    }


def leather_like_casual_preferred(bgr: np.ndarray, cfg: dict | None = None) -> bool:
    """
    True when the (preprocessed) frame looks like leather / dress / muted casual footwear
    rather than a colorful athletic trainer. Used to correct histogram + rule bias toward
    "sports" on brown/black/navy leather and white-sole dress shoes.
    """
    if bgr is None or not hasattr(bgr, "shape") or bgr.size == 0:
        return False
    cfg = cfg or load_config()
    vm = cfg.get("vision", {})
    if not bool(vm.get("leather_like_detection_enabled", True)):
        return False

    mask = _segment_shoe_mask(bgr)
    color = _compute_color_features_masked(bgr, mask)
    edge_tex = _compute_edge_texture_masked(bgr, mask)

    sm = color["sat_mean"]
    br = color["bright_ratio"]
    cv_val = color["color_variety"]
    val_mean = color["val_mean"]
    ed = edge_tex["edge_density"]
    gm = edge_tex["gradient_mean"]
    tr = edge_tex["texture_rms"]

    # Neon / multi-panel trainers — do not treat as leather
    if sm >= float(vm.get("leather_like_block_sat_min", 0.36)) and cv_val >= float(
        vm.get("leather_like_block_variety_min", 8.0)
    ):
        return False
    if ed >= float(vm.get("leather_like_block_edge_min", 0.26)) and gm >= float(
        vm.get("leather_like_block_grad_min", 0.16)
    ):
        return False

    sat_max = float(vm.get("leather_like_sat_mean_max", 0.30))
    br_max = float(vm.get("leather_like_bright_ratio_max", 0.34))
    cv_max = float(vm.get("leather_like_color_variety_max", 6.8))

    muted = sm < sat_max and br < br_max and cv_val <= cv_max
    dark_leather = sm < float(vm.get("leather_like_dark_sat_max", 0.22)) and val_mean < float(
        vm.get("leather_like_dark_val_mean_max", 0.52)
    )
    pebble = (
        float(vm.get("leather_like_pebble_texture_min", 0.040))
        <= tr
        <= float(vm.get("leather_like_pebble_texture_max", 0.14))
        and sm < float(vm.get("leather_like_pebble_sat_max", 0.33))
        and gm < float(vm.get("leather_like_pebble_grad_max", 0.128))
    )
    # Navy / oxblood leather: slightly higher S but still not "trainer vivid"
    navy = (
        sm < float(vm.get("leather_like_navy_sat_max", 0.34))
        and val_mean < float(vm.get("leather_like_navy_val_mean_max", 0.48))
        and br < float(vm.get("leather_like_navy_bright_ratio_max", 0.26))
        and cv_val <= float(vm.get("leather_like_navy_variety_max", 6.2))
    )

    return bool(muted or dark_leather or pebble or navy)


def leather_like_strong_casual_override(bgr: np.ndarray, cfg: dict | None = None) -> bool:
    """Very muted upper — almost always leather/dress/casual, not a vivid trainer."""
    if bgr is None or not hasattr(bgr, "shape") or bgr.size == 0:
        return False
    cfg = cfg or load_config()
    vm = cfg.get("vision", {})
    if not bool(vm.get("leather_like_detection_enabled", True)):
        return False
    mask = _segment_shoe_mask(bgr)
    color = _compute_color_features_masked(bgr, mask)
    edge_tex = _compute_edge_texture_masked(bgr, mask)
    sm = color["sat_mean"]
    br = color["bright_ratio"]
    ed = edge_tex["edge_density"]
    gm = edge_tex["gradient_mean"]
    if ed >= float(vm.get("leather_like_strong_block_edge_min", 0.24)) and gm >= float(
        vm.get("leather_like_strong_block_grad_min", 0.148)
    ):
        return False
    return sm < float(vm.get("leather_like_strong_sat_max", 0.188)) and br < float(
        vm.get("leather_like_strong_bright_max", 0.34)
    )


def classify_shoe_type_rule_based(bgr: np.ndarray, cfg: dict) -> tuple[ShoeCategory, dict[str, float]]:
    """
    Accurate sports vs casual using foreground segmentation + multi-feature analysis.
    Features are computed only on the shoe (background excluded):
      - Shape: aspect ratio, sole thickness, contour complexity, solidity
      - Texture: LBP (mesh detection), edge density, gradient, high-freq energy
      - Color: saturation, brightness, hue variety (on shoe only)
      - Structure: sole/upper contrast (thick bright soles = sports)
    """
    vm = cfg.get("vision", {})

    mask = _segment_shoe_mask(bgr)
    shape = _extract_shoe_shape_features(mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = _compute_lbp_texture(gray, mask)
    color = _compute_color_features_masked(bgr, mask)
    edge_tex = _compute_edge_texture_masked(bgr, mask)
    sole = _compute_sole_contrast(bgr, mask)
    h_profile = _analyze_horizontal_profile(bgr)

    sports_score = 0.0
    casual_score = 0.0

    # ===== 1. SOLE CONTRAST (strongest signal — booth lighting makes this reliable) =====
    sbd = sole["sole_brightness_diff"]
    scd = sole["sole_color_diff"]
    # Sports shoes almost always have a distinct (often white) midsole contrasting upper
    if sbd >= 0.12 or scd >= 0.10:
        sports_score += 0.20
    elif sbd >= 0.06 or scd >= 0.05:
        sports_score += 0.10
    # Very low contrast = uniform material = casual/leather
    if sbd < 0.03 and scd < 0.03:
        casual_score += 0.12

    # ===== 2. HORIZONTAL PROFILE (sole zone edge pattern) =====
    sep = h_profile["sole_edge_prominence"]
    sgb = h_profile["sole_grad_boost"]
    sld = h_profile["sole_line_density"]

    # Chunky treaded sole creates a strong edge band in bottom strips
    if sep >= 0.04:
        sports_score += 0.12
    elif sep >= 0.02:
        sports_score += 0.06
    # Top-heavy edges (laces/collar) with clean bottom = flat sole = casual
    if sep < -0.02:
        casual_score += 0.08

    if sgb >= 0.03:
        sports_score += 0.08
    elif sgb < -0.02:
        casual_score += 0.04

    if sld >= 0.06:
        sports_score += 0.04

    # ===== 3. COLOR ANALYSIS (moderate — works on booth images) =====
    cv_val = color["color_variety"]
    sm = color["sat_mean"]
    br = color["bright_ratio"]

    # Multi-color panels (mesh overlays, swoosh, stripes) = sports
    if cv_val >= 7.0 and sm >= 0.20:
        sports_score += 0.10
    elif cv_val >= 5.0 and sm >= 0.15:
        sports_score += 0.05
    # Uniform muted single color = casual/leather
    if cv_val <= 3.0 and sm < 0.15:
        casual_score += 0.10
    elif cv_val <= 4.0 and sm < 0.10:
        casual_score += 0.05

    # Bright white areas (midsole foam, stripes) common in athletic shoes
    if br >= 0.25:
        sports_score += 0.06
    elif br < 0.05:
        casual_score += 0.04

    # ===== 4. SHAPE FEATURES (from segmented mask) =====
    wv = shape["width_variation"]
    nd = shape["convexity_defects"]
    cb = shape["collar_bulge"]
    cx = shape["complexity"]
    st = shape["sole_thickness_norm"]
    sr = shape["sole_ratio"]

    # Width variation: sports have flared sole vs narrower upper
    if wv >= 0.30:
        sports_score += 0.08
    elif wv < 0.10:
        casual_score += 0.04

    # Many convexity defects = complex multi-panel construction
    if nd >= 15:
        sports_score += 0.06
    elif nd < 3:
        casual_score += 0.04

    # Padded collar bulge
    if cb >= 0.12:
        sports_score += 0.04

    # ===== 5. TEXTURE (lowest weight — dirt-sensitive) =====
    ed = edge_tex["edge_density"]
    gm = edge_tex["gradient_mean"]
    hfe = edge_tex["high_freq_energy"]

    # Very high edge + gradient = mesh/knit material
    if ed >= 0.25 and gm >= 0.15:
        sports_score += 0.06
    # Very smooth = leather/canvas
    if ed < 0.05 and gm < 0.05:
        casual_score += 0.06

    # Leather / dress / muted casual — strong push away from "sports" (histogram + sole cues
    # often misread brogue stitch + rubber sole as athletic).
    leather_like = leather_like_casual_preferred(bgr, cfg)
    if leather_like:
        casual_score += float(vm.get("leather_like_rule_casual_boost", 0.26))
        sports_score *= float(vm.get("leather_like_rule_sports_scale", 0.42))
        # Chunky-sole heuristics are misleading on leather oxfords; dampen them.
        sports_score -= float(vm.get("leather_like_rule_sole_contrast_penalty", 0.06))

    # Final fusion
    raw_diff = sports_score - casual_score
    fusion = max(0.0, min(1.0, 0.5 + raw_diff))

    thr = float(vm.get("sports_fusion_threshold", 0.52))
    cat = ShoeCategory.SPORTS if fusion >= thr else ShoeCategory.CASUAL

    metrics: dict[str, float] = {
        "edge_density": ed,
        "gradient_mean": gm,
        "texture_rms": edge_tex["texture_rms"],
        "saturation_mean": color["sat_mean"] * 255.0,
        "sports_fusion_score": fusion,
        "sports_fusion_threshold": thr,
        "shape_aspect_ratio": shape["aspect_ratio"],
        "shape_sole_ratio": shape["sole_ratio"],
        "shape_sole_thickness": shape["sole_thickness_norm"],
        "shape_complexity": shape["complexity"],
        "shape_solidity": shape["solidity"],
        "shape_width_variation": shape["width_variation"],
        "shape_collar_bulge": shape["collar_bulge"],
        "shape_convexity_defects": shape["convexity_defects"],
        "lbp_std": lbp["lbp_std"],
        "lbp_uniformity": lbp["lbp_uniformity"],
        "color_variety": color["color_variety"],
        "bright_ratio": color["bright_ratio"],
        "sole_brightness_diff": sole["sole_brightness_diff"],
        "sole_color_diff": sole["sole_color_diff"],
        "high_freq_energy": hfe,
        "sole_edge_prominence": sep,
        "sole_grad_boost": sgb,
        "sole_line_density": sld,
        "sports_evidence": sports_score,
        "casual_evidence": casual_score,
        "leather_like_casual": 1.0 if leather_like else 0.0,
    }

    log.debug(
        "shoe_type: %s (fusion=%.3f thr=%.2f) sport=%.3f casual=%.3f "
        "sole_br=%.3f sole_cd=%.3f sep=%.3f sgb=%.3f "
        "sat=%.3f cv=%.1f br=%.3f wv=%.3f nd=%.0f",
        cat.value, fusion, thr, sports_score, casual_score,
        sbd, scd, sep, sgb, sm, cv_val, br, wv, nd,
    )

    return cat, metrics


def _category_heuristic(bgr: np.ndarray) -> ShoeCategory:
    """
    Sports vs leather vs casual from frame statistics.
    Pebbled / grain leather raises texture_rms and grad — handled explicitly so it is not misread as mesh sports.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blur, 50, 150)
    edge_d = float(np.mean(edges > 0))

    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mean = float(np.mean(np.sqrt(gx * gx + gy * gy))) / 255.0

    # High-frequency residual = canvas / dirty textile vs smooth vs pebbled leather
    bl5 = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - bl5.astype(np.float32)
    texture_rms = float(np.sqrt(np.mean(residual**2))) / 255.0

    cfg = load_config()
    vm = cfg.get("vision", {})
    ls_max = float(vm.get("leather_saturation_max", 52))
    le_max = float(vm.get("leather_edge_max", 0.092))
    lg_max = float(vm.get("leather_gradient_max", 0.086))
    se_min = float(vm.get("sports_edge_min", 0.072))
    sg_min = float(vm.get("sports_gradient_min", 0.09))
    # Mesh trainers produce higher global gradient than smooth leather + sole seam (often ~0.10–0.12)
    mesh_grad_min = float(vm.get("sports_mesh_gradient_min", 0.130))
    bright_s = float(vm.get("sports_bright_saturation_min", 70))
    bright_e = float(vm.get("sports_bright_edge_min", 0.05))

    # Muted saturation (some blacks/grays) + sole contrast — not mesh
    sat_muted = s < float(vm.get("leather_muted_saturation_max", 66))
    # Raw high gradient (brogue/stitch can look “mesh-like” — do not use alone for sports)
    mesh_texture = grad_mean >= mesh_grad_min
    leather_sole_contrast = (
        sat_muted
        and not mesh_texture
        and edge_d >= float(vm.get("leather_min_edge_when_muted", 0.042))
    )

    sat_for_edge_sports = float(vm.get("sports_edges_require_saturation_min", 60))
    min_grad_for_color_sports = float(vm.get("sports_edge_color_min_gradient", 0.126))
    edge_plus_color = (
        edge_d >= se_min
        and s >= sat_for_edge_sports
        and grad_mean >= min_grad_for_color_sports
    )
    bright_sports = s >= bright_s and edge_d >= bright_e

    v_leather_max = float(vm.get("leather_brightness_v_max", 108))
    leather_tex_max = float(vm.get("leather_max_texture_rms", 0.072))
    leather_sole_tex_max = float(vm.get("leather_sole_max_texture_rms", 0.10))
    # Dark upper: brogue/stitch can raise mean gradient — still leather, not knit
    leather_dark_smooth = (
        v_mean < v_leather_max
        and edge_d >= float(vm.get("leather_min_edge_when_muted", 0.042))
        and texture_rms < leather_tex_max
        and not bright_sports
        and not edge_plus_color
    )

    # “Leather-like” smooth — exclude athletic cues
    leather_like = (
        s < ls_max
        and edge_d <= le_max
        and grad_mean <= lg_max
        and not mesh_texture
        and not edge_plus_color
        and not bright_sports
    )

    leather_safe_grad = float(vm.get("leather_safe_gradient_max", 0.132))
    leather_smooth_upper = (
        not mesh_texture
        and grad_mean < leather_safe_grad
        and edge_d >= float(vm.get("leather_min_edge_when_muted", 0.042))
        and not bright_sports
        and not edge_plus_color
        and texture_rms < leather_tex_max
    )

    casual_tex_min = float(vm.get("casual_texture_rms_min", 0.048))
    casual_fabric = (
        texture_rms >= casual_tex_min
        and not mesh_texture
        and grad_mean < mesh_grad_min
        and edge_d >= float(vm.get("leather_min_edge_when_muted", 0.038))
        and not bright_sports
        and not edge_plus_color
    )

    # Bright, smooth, low saturation (white PU / “dress sneaker”): not natural leather grain
    casual_synthetic_v_min = float(vm.get("casual_synthetic_brightness_v_min", 150))
    casual_synthetic_s_max = float(vm.get("casual_synthetic_saturation_max", 58))
    casual_synthetic_tex_max = float(vm.get("casual_synthetic_texture_rms_max", 0.062))
    casual_synthetic_grad_max = float(vm.get("casual_synthetic_gradient_max", 0.125))
    casual_synthetic_smooth = (
        v_mean >= casual_synthetic_v_min
        and s <= casual_synthetic_s_max
        and texture_rms < casual_synthetic_tex_max
        and not mesh_texture
        and not edge_plus_color
        and not bright_sports
        and grad_mean < casual_synthetic_grad_max
    )

    # Knit/mesh trainer: require mesh + (bright OR grad+sat OR very high grad on low-sat gray knit).
    # Leather grain used to satisfy “grad OR sat” alone and was misread as sports.
    mesh_strong_grad = float(vm.get("sports_mesh_strong_gradient_min", 0.138))
    mesh_min_sat = float(vm.get("sports_mesh_min_saturation", 36))
    low_sat_grad = float(vm.get("sports_mesh_low_sat_gradient_min", 0.148))
    low_sat_min = float(vm.get("sports_mesh_low_sat_min", 18))
    sports_mesh_knit = mesh_texture and (
        bright_sports
        or (grad_mean >= mesh_strong_grad and s >= mesh_min_sat)
        or (grad_mean >= low_sat_grad and s >= low_sat_min)
    )

    strong_sports = sports_mesh_knit or edge_plus_color or bright_sports

    # “Obvious” trainer: colorful, bright, or strong knit — only then trust sports.
    sports_obvious_color_sat = float(vm.get("sports_obvious_color_sat_min", 64))
    sports_obvious_mesh_grad = float(vm.get("sports_obvious_mesh_grad_min", 0.144))
    sports_obvious_mesh_sat = float(vm.get("sports_obvious_mesh_sat_min", 42))
    sports_obvious = (
        bright_sports
        or (edge_plus_color and s >= sports_obvious_color_sat)
        or (
            mesh_texture
            and grad_mean >= sports_obvious_mesh_grad
            and s >= sports_obvious_mesh_sat
        )
    )
    dress_ambiguous_max_tex = float(vm.get("dress_ambiguous_max_texture_rms", 0.102))
    dress_ambiguous_max_v = float(vm.get("dress_ambiguous_brightness_v_max", 132))
    dress_ambiguous = (
        sat_muted
        and not bright_sports
        and texture_rms < dress_ambiguous_max_tex
        and v_mean < dress_ambiguous_max_v
    )

    # Pebbled / tumbled leather (loafers, boat shoes): high texture_rms + grad trips “mesh”
    # but saturation stays muted — classify as leather, not sports knit.
    lp_tex_min = float(vm.get("leather_pebble_texture_rms_min", 0.048))
    lp_tex_max = float(vm.get("leather_pebble_texture_rms_max", 0.175))
    lp_v_max = float(vm.get("leather_pebble_brightness_v_max", 148))
    lp_s_max = float(vm.get("leather_pebble_saturation_max", 64))
    leather_pebbled_grain = (
        sat_muted
        and not bright_sports
        and not edge_plus_color
        and texture_rms >= lp_tex_min
        and texture_rms <= lp_tex_max
        and v_mean < lp_v_max
        and s < lp_s_max
    )

    # Muted leather (dress/boots): before sports — stitch/brogue must not lose to “mesh”
    muted_leather_scene = sat_muted and not edge_plus_color and not bright_sports
    if muted_leather_scene:
        if casual_synthetic_smooth:
            return ShoeCategory.CASUAL
        if leather_pebbled_grain:
            return ShoeCategory.CASUAL
        if leather_like:
            return ShoeCategory.CASUAL
        if leather_sole_contrast and texture_rms < leather_sole_tex_max:
            return ShoeCategory.CASUAL
        if leather_dark_smooth:
            return ShoeCategory.CASUAL
        if leather_smooth_upper:
            return ShoeCategory.CASUAL

    if strong_sports:
        if leather_pebbled_grain:
            return ShoeCategory.CASUAL
        # Prefer casual over sports when the scene looks dress/leather-adjacent but
        # framing/lighting triggered weak athletic cues (not a clear trainer).
        if dress_ambiguous and not sports_obvious:
            return ShoeCategory.CASUAL
        return ShoeCategory.SPORTS
    if casual_synthetic_smooth:
        return ShoeCategory.CASUAL
    if leather_like:
        return ShoeCategory.CASUAL
    if leather_sole_contrast and texture_rms < leather_sole_tex_max:
        return ShoeCategory.CASUAL
    if leather_dark_smooth:
        return ShoeCategory.CASUAL
    if leather_smooth_upper:
        return ShoeCategory.CASUAL
    if casual_fabric:
        return ShoeCategory.CASUAL
    return ShoeCategory.CASUAL


def encode_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes() if ok else None


def blank_jpeg_bytes(width: int = 640, height: int = 480) -> bytes:
    img = np.zeros((max(2, height), max(2, width), 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b""


def analyze_frame(bgr: np.ndarray, cfg: dict | None = None) -> VisionResult:
    cfg = cfg or load_config()
    use_rules = bool(cfg.get("vision", {}).get("rule_based_pipeline", False))
    if use_rules:
        cat, m = classify_shoe_type_rule_based(bgr, cfg)
        d_score, d_level, dirty_r = _rule_dirt_level_and_score(bgr, cfg)
        return VisionResult(
            dirt_score=d_score,
            category=cat,
            frame_bgr=bgr,
            dirt_level=d_level,
            edge_density=m["edge_density"],
            dirty_pixel_ratio=dirty_r,
            sports_fusion_score=m["sports_fusion_score"],
            sports_fusion_threshold=m["sports_fusion_threshold"],
            gradient_mean=m["gradient_mean"],
            texture_rms=m["texture_rms"],
            saturation_mean=m["saturation_mean"],
            leather_like_casual=m.get("leather_like_casual"),
        )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    d = _dirt_score(gray)
    cat = _category_heuristic(bgr)
    return VisionResult(dirt_score=d, category=cat, frame_bgr=bgr)


class WebcamCapture:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        c = self.cfg["camera"]
        self.index = int(c["index"])
        self.w = int(c["width"])
        self.h = int(c["height"])
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._cap is None:
                self.open()
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.error("webcam read failed")
                return None
            return frame

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
