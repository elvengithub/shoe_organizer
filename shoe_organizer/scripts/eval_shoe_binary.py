#!/usr/bin/env python3
"""
Evaluate models/shoe_binary.tflite on image folders (same preprocessing as Pi runtime).

  pip install -r scripts/requirements-train.txt
  python scripts/eval_shoe_binary.py --pos datasets/shoe_binary/shoe --neg datasets/shoe_binary/not_shoe
  python scripts/eval_shoe_binary.py --align_preprocess --threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".tif", ".tiff"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def collect(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _EXT
    ]


def load_bgr(path: Path) -> np.ndarray | None:
    import cv2
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            rgb = np.asarray(im, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/shoe_binary.tflite")
    parser.add_argument("--pos", nargs="+", default=["datasets/shoe_binary/shoe", "datasets/shoes"])
    parser.add_argument("--neg", nargs="+", default=["datasets/shoe_binary/not_shoe", "datasets/not_shoe"])
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--align_preprocess", action="store_true")
    args = parser.parse_args()

    base = _repo_root()
    sys.path.insert(0, str(base))

    model_path = (base / args.model).resolve()
    if not model_path.is_file():
        raise SystemExit(f"Missing model: {model_path}")

    cfg = None
    if args.align_preprocess:
        import yaml

        with open(base / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    from src.shoe_binary_tflite import predict_p_shoe

    items: list[tuple[Path, int]] = []
    for d in args.pos:
        for p in collect((base / d).resolve()):
            items.append((p, 1))
    for d in args.neg:
        for p in collect((base / d).resolve()):
            items.append((p, 0))

    seen: set[str] = set()
    unique: list[tuple[Path, int]] = []
    for p, y in items:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        unique.append((p, y))

    if len(unique) < 2:
        raise SystemExit("Need at least 2 images total across pos/neg folders.")

    sb = {"input_size": args.size, "threshold": args.threshold}

    probs: list[float] = []
    labels: list[int] = []
    for p, y in unique:
        bgr = load_bgr(p)
        if bgr is None:
            continue
        if cfg is not None:
            from src.vision_preprocess import apply_vision_preprocess

            bgr = apply_vision_preprocess(bgr, cfg)
        p_shoe = predict_p_shoe(bgr, str(model_path), sb)
        if p_shoe is None:
            continue
        probs.append(p_shoe)
        labels.append(y)

    probs_a = np.array(probs, dtype=np.float32)
    labels_a = np.array(labels, dtype=np.int32)
    pred = (probs_a >= args.threshold).astype(np.int32)
    tp = int(np.sum((pred == 1) & (labels_a == 1)))
    tn = int(np.sum((pred == 0) & (labels_a == 0)))
    fp = int(np.sum((pred == 1) & (labels_a == 0)))
    fn = int(np.sum((pred == 0) & (labels_a == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    out = {
        "n_images": len(probs_a),
        "threshold": args.threshold,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision_shoe": round(prec, 4),
        "recall_shoe": round(rec, 4),
        "f1_shoe": round(f1, 4),
        "accuracy": round(acc, 4),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
