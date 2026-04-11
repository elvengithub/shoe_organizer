"""
ONNX shoe object detector — same flow as jmaloney95/Shoe-ObjectDetection-and-ImageClassification
(detect-shoe.py / deployed_model.py): ImageNet-normalized 640×640 crop → ONNX → best box.

Optional: only runs when config shoe_object_detection.enabled and onnx_path points to a local .onnx file.
Requires: pip install onnxruntime onnx
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .config_loader import load_config

log = logging.getLogger(__name__)

_ort_session = None
_ort_path: str | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_onnx_path(cfg: dict) -> Path | None:
    block = cfg.get("shoe_object_detection") or {}
    raw = str(block.get("onnx_path") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = _project_root() / p
    return p if p.is_file() else None


def _get_session(onnx_path: Path):
    global _ort_session, _ort_path
    key = str(onnx_path.resolve())
    if _ort_session is not None and _ort_path == key:
        return _ort_session
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("shoe_object_detection: onnxruntime not installed — pip install onnxruntime onnx")
        return None
    _ort_session = ort.InferenceSession(
        key,
        providers=["CPUExecutionProvider"],
    )
    _ort_path = key
    return _ort_session


def _preprocess_bgr(bgr: np.ndarray, input_size: int) -> tuple[np.ndarray, float, float]:
    """Resize (shortest edge) → center crop → RGB ImageNet norm. Returns NCHW float32 batch + scale to map boxes to original."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    scale = input_size / min(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0 = max(0, (nh - input_size) // 2)
    x0 = max(0, (nw - input_size) // 2)
    crop = resized[y0 : y0 + input_size, x0 : x0 + input_size]
    if crop.shape[0] != input_size or crop.shape[1] != input_size:
        crop = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std
    chw = np.transpose(crop, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0).astype(np.float32)
    wr = w / float(input_size)
    hr = h / float(input_size)
    return batch, wr, hr


def _parse_detections(raw_out, conf_thr: float) -> list[dict]:
    """Match detect-shoe.py process_output."""
    arr = np.asarray(raw_out[0])
    if arr.ndim == 3:
        detections = arr[0]
    elif arr.ndim == 2:
        detections = arr
    else:
        return []
    boxes: list[dict] = []
    for detection in detections:
        if detection.size < 6:
            continue
        conf = float(detection[1])
        if conf <= conf_thr:
            continue
        boxes.append(
            {
                "class_id": int(detection[0]),
                "confidence": conf,
                "x": int(detection[2]),
                "y": int(detection[3]),
                "width": abs(int(detection[4] - detection[2])),
                "height": abs(int(detection[5] - detection[3])),
            }
        )
    if not boxes:
        return []
    return [max(boxes, key=lambda x: x["confidence"])]


def try_onnx_shoe_gate(bgr: np.ndarray, cfg: dict) -> "ShoeGateResult | None":
    """
    If ONNX shoe detection is configured and available, return ShoeGateResult.
    Return None to fall back to classic OpenCV contour gate (missing model, import error, etc.).
    """
    from .vision_service import ShoeGateResult

    block = cfg.get("shoe_object_detection") or {}
    if not bool(block.get("enabled", False)):
        return None

    path = _resolve_onnx_path(cfg)
    if path is None:
        log.debug("shoe_object_detection: onnx_path missing or file not found — using contour gate")
        return None

    session = _get_session(path)
    if session is None:
        return None

    input_size = int(block.get("input_size", 640))
    conf_thr = float(block.get("confidence_threshold", 0.5))
    min_area_ratio = float(block.get("min_box_area_ratio", 0.015))

    inp = session.get_inputs()[0]
    input_name = str(block.get("input_name") or inp.name)

    try:
        batch, wr, hr = _preprocess_bgr(bgr, input_size)
        out = session.run(None, {input_name: batch})
        boxes = _parse_detections(out, conf_thr)
    except Exception as e:
        log.warning("shoe_object_detection inference failed: %s", e)
        return None

    if not boxes:
        return ShoeGateResult(False, "od_no_detection")

    box = boxes[0]
    ih, iw = bgr.shape[:2]
    x1 = int(box["x"] * wr)
    y1 = int(box["y"] * hr)
    x2 = x1 + int(box["width"] * wr)
    y2 = y1 + int(box["height"] * hr)
    x1 = max(0, min(iw - 1, x1))
    y1 = max(0, min(ih - 1, y1))
    x2 = max(x1 + 1, min(iw, x2))
    y2 = max(y1 + 1, min(ih, y2))
    area = float((x2 - x1) * (y2 - y1))
    if area / max(1.0, float(iw * ih)) < min_area_ratio:
        return ShoeGateResult(False, "od_box_too_small")

    return ShoeGateResult(True, "od_shoe")


def onnx_gate_configured_and_ready() -> bool:
    """True if enabled, file exists, and onnxruntime loads (for tests / UI hints)."""
    cfg = load_config()
    block = cfg.get("shoe_object_detection") or {}
    if not bool(block.get("enabled", False)):
        return False
    p = _resolve_onnx_path(cfg)
    if p is None:
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _get_session(p) is not None
