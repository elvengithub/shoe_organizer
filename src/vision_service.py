from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .config_loader import load_config

log = logging.getLogger(__name__)


class ShoeCategory(str, Enum):
    """Camera-only bucket: sports | leather | casual (three-way)."""

    LEATHER = "leather"
    SPORTS = "sports"
    CASUAL = "casual"


@dataclass
class VisionResult:
    dirt_score: float
    category: ShoeCategory
    frame_bgr: np.ndarray | None = None


@dataclass
class ShoeGateResult:
    """OpenCV silhouette heuristics — tune in config `shoe_gate`. Not a full classifier."""

    is_shoe: bool
    reason: str


def evaluate_shoe_gate(bgr: np.ndarray, cfg: dict | None = None) -> ShoeGateResult:
    cfg = cfg or load_config()
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
            return ShoeCategory.LEATHER
        if leather_like:
            return ShoeCategory.LEATHER
        if leather_sole_contrast and texture_rms < leather_sole_tex_max:
            return ShoeCategory.LEATHER
        if leather_dark_smooth:
            return ShoeCategory.LEATHER
        if leather_smooth_upper:
            return ShoeCategory.LEATHER

    if strong_sports:
        if leather_pebbled_grain:
            return ShoeCategory.LEATHER
        # Prefer casual over sports when the scene looks dress/leather-adjacent but
        # framing/lighting triggered weak athletic cues (not a clear trainer).
        if dress_ambiguous and not sports_obvious:
            return ShoeCategory.CASUAL
        return ShoeCategory.SPORTS
    if casual_synthetic_smooth:
        return ShoeCategory.CASUAL
    if leather_like:
        return ShoeCategory.LEATHER
    if leather_sole_contrast and texture_rms < leather_sole_tex_max:
        return ShoeCategory.LEATHER
    if leather_dark_smooth:
        return ShoeCategory.LEATHER
    if leather_smooth_upper:
        return ShoeCategory.LEATHER
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


def analyze_frame(bgr: np.ndarray) -> VisionResult:
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
