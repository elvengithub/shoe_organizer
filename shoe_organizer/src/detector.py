"""
Stage 1: Ultralytics YOLOv8 shoe detection — bounding boxes + crop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ShoeDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    crop_bgr: np.ndarray


class ShoeDetector:
    """Loads a YOLOv8 `.pt` model and returns the highest-confidence shoe box per frame."""

    def __init__(
        self,
        weights: Path,
        *,
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 416,
        pad_ratio: float = 0.02,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.pad_ratio = float(pad_ratio)

    def detect_best(self, bgr: np.ndarray) -> ShoeDetection | None:
        if bgr is None or bgr.size == 0:
            return None
        res = self._model.predict(
            source=bgr,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not res:
            return None
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        confs = boxes.conf.cpu().numpy()
        best_i = int(np.argmax(confs))
        xyxy = boxes.xyxy[best_i].cpu().numpy().astype(np.float64)
        conf = float(confs[best_i])
        x1, y1, x2, y2 = map(float, xyxy.tolist())
        crop = self._crop_padded(bgr, x1, y1, x2, y2)
        if crop is None or crop.size == 0:
            return None
        return ShoeDetection(xyxy=(x1, y1, x2, y2), confidence=conf, crop_bgr=crop)

    def _crop_padded(self, bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * self.pad_ratio
        pad_y = bh * self.pad_ratio
        nx1 = int(max(0, np.floor(x1 - pad_x)))
        ny1 = int(max(0, np.floor(y1 - pad_y)))
        nx2 = int(min(w, np.ceil(x2 + pad_x)))
        ny2 = int(min(h, np.ceil(y2 + pad_y)))
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return bgr[ny1:ny2, nx1:nx2].copy()
