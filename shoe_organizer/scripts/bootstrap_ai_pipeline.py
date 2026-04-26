#!/usr/bin/env python3
"""
One-shot local setup: synthetic dataset, train both classifier heads, download YOLOv8n
as models/shoe_detector.pt (for when you set prefer_full_roi: false).

Expects ai_pipeline.enabled / prefer_full_roi already set in config.yaml (see repo default).

Run from the ``shoe_organizer`` app directory::

  pip install -r requirements.txt -r requirements-ai.txt
  python scripts/bootstrap_ai_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
TYPE_CLASSES = ["sneaker", "boot", "sandal", "loafer", "other"]
CLEAN_CLASSES = ["clean", "dirty"]


def _write_synthetic_images() -> None:
    rng = np.random.default_rng(42)
    base = _ROOT / "dataset"

    for label, n_img, dirty in [("clean", 64, False), ("dirty", 64, True)]:
        d = base / "cleanliness" / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_img):
            h, w = 256, 256
            if dirty:
                img = rng.integers(40, 200, (h, w, 3), dtype=np.uint8)
                img = np.clip(
                    img.astype(np.float32) + rng.normal(0, 42, (h, w, 3)),
                    0,
                    255,
                ).astype(np.uint8)
                img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + 35, 0, 255).astype(np.uint8)
                img = cv2.GaussianBlur(img, (3, 3), 0)
            else:
                v = int(rng.integers(110, 180))
                img = np.full((h, w, 3), v, dtype=np.uint8)
                cv2.GaussianBlur(img, (0, 0), sigmaX=2.0, dst=img)
            cv2.imwrite(str(d / f"{label}_{i:03d}.jpg"), img)

    profiles = {
        "sneaker": ((rng.normal(0, 18, 3) + np.array([40, 90, 200])).clip(0, 255), 28.0),
        "boot": ((rng.normal(0, 12, 3) + np.array([55, 55, 70])).clip(0, 255), 12.0),
        "sandal": ((rng.normal(0, 15, 3) + np.array([180, 160, 140])).clip(0, 255), 18.0),
        "loafer": ((rng.normal(0, 10, 3) + np.array([30, 30, 90])).clip(0, 255), 10.0),
        "other": ((rng.normal(0, 20, 3) + np.array([100, 100, 100])).clip(0, 255), 22.0),
    }
    for cls, (mean_bgr, noise) in profiles.items():
        d = base / "type" / cls
        d.mkdir(parents=True, exist_ok=True)
        mean_bgr = mean_bgr.astype(np.float32)
        for i in range(48):
            h, w = 256, 256
            img = rng.normal(mean_bgr, noise, (h, w, 3)).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"{cls}_{i:03d}.jpg"), img)


def _download_yolov8n() -> Path:
    dest = _ROOT / "models" / "shoe_detector.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    ]
    last_err: Exception | None = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download yolov8n.pt: {last_err}") from last_err


def _run_train(args: list[str]) -> None:
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "train_torch_classifier.py"), *args],
        cwd=str(_ROOT),
    )
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    print("1/4 Synthetic images...")
    _write_synthetic_images()
    print("2/4 Train cleanliness head...")
    _run_train(
        [
            "--data",
            str(_ROOT / "dataset" / "cleanliness"),
            "--out",
            str(_ROOT / "models" / "cleanliness_classifier.pt"),
            "--classes",
            ",".join(CLEAN_CLASSES),
            "--epochs",
            "8",
            "--batch",
            "16",
        ]
    )
    print("3/4 Train type head...")
    _run_train(
        [
            "--data",
            str(_ROOT / "dataset" / "type"),
            "--out",
            str(_ROOT / "models" / "shoe_type_classifier.pt"),
            "--classes",
            ",".join(TYPE_CLASSES),
            "--epochs",
            "10",
            "--batch",
            "16",
        ]
    )
    print("4/4 YOLOv8n -> models/shoe_detector.pt ...")
    path = _download_yolov8n()
    print("  saved", path)
    print("Done. config: ai_pipeline.enabled + prefer_full_roi (see config.yaml).")
    print("  Demo: python -m src.main   |   App: python run.py")
    print("  Replace synthetic data with real photos; set prefer_full_roi: false when using a shoe YOLO.")


if __name__ == "__main__":
    main()
