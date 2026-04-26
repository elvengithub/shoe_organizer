"""
Standalone real-time demo: capture → YOLO crop → type + cleanliness classifiers → OpenCV overlay.

Run from the `shoe_organizer` app directory:
  pip install -r requirements.txt -r requirements-ai.txt
  Set ``ai_pipeline.enabled: true`` and place weights under ``models/``.
  python -m src.main
"""
from __future__ import annotations

import logging
import os
import sys

# Allow `python -m src.main` from the app folder (sibling of src)
_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import cv2

from src.config_loader import load_config
from src.camera import NormalizedCamera
from src.two_stage_vision import infer_two_stage
from src.vision_preprocess import apply_vision_preprocess

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    ap = cfg.get("ai_pipeline") or {}
    if not bool(ap.get("enabled")):
        log.warning("ai_pipeline.enabled is false — enable it in config.yaml and add weights under models/.")

    cam = NormalizedCamera(cfg)
    cam.open()
    win = "shoe_organizer — two-stage pipeline (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            bgr = apply_vision_preprocess(frame, cfg)
            vision, dbg = infer_two_stage(bgr, cfg)

            # Draw on the same ROI the models see (boxes are in preprocessed coordinates).
            display = bgr.copy()
            if vision and vision.detector_box_xyxy:
                x1, y1, x2, y2 = map(int, vision.detector_box_xyxy)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if vision:
                ft = vision.fine_shoe_type or "?"
                cl = (vision.dirt_level or "?").replace("_", " ")
                tconf = vision.shoe_type_confidence or 0.0
                cconf = vision.cleanliness_confidence or 0.0
                lines = [
                    f"Type: {ft} ({tconf:.2f})",
                    f"Cleanliness: {cl} ({cconf:.2f})",
                    f"Bucket: {vision.category.value}",
                ]
            else:
                lines = [
                    str(dbg.get("ai_pipeline_error", "ai_pipeline disabled or weights missing")),
                ]
            y0 = 24
            for i, line in enumerate(lines):
                cv2.putText(
                    display,
                    line,
                    (12, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255) if vision else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if dbg.get("inference_ms") is not None:
                cv2.putText(
                    display,
                    f"{dbg['inference_ms']} ms",
                    (12, display.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
