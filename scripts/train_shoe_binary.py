#!/usr/bin/env python3
"""
Train a binary shoe classifier (MobileNetV2 head) and export models/shoe_binary.tflite.

  pip install -r scripts/requirements-train.txt
  python scripts/train_shoe_binary.py
  python scripts/train_shoe_binary.py --align_preprocess --epochs 20 --finetune_epochs 4

Data (defaults merge all of these):
  - Positives: datasets/shoe_binary/shoe, datasets/shoes (catalog tree), datasets/dirty_Shoe
  - Negatives: datasets/shoe_binary/not_shoe, datasets/not_shoe

--align_preprocess: apply config.yaml vision_preprocess (ROI + CLAHE) like production inference.
Normalization matches runtime (src/shoe_binary_tflite.py): rgb / 127.5 - 1.0

Outputs:
  - models/shoe_binary.tflite
  - models/shoe_binary_metrics.json  (confusion, precision/recall/F1, suggested threshold)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".tif", ".tiff"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def collect(root: Path, label: int) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _EXT:
            out.append((p, label))
    return out


def merge_unique(items: list[tuple[Path, int]]) -> list[tuple[Path, int]]:
    seen: set[str] = set()
    out: list[tuple[Path, int]] = []
    for p, y in items:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        out.append((p, y))
    return out


def stratified_split(
    pos: list[tuple[Path, int]],
    neg: list[tuple[Path, int]],
    val_frac: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)

    def split_one(items: list[tuple[Path, int]]) -> tuple[list, list]:
        items = items.copy()
        rng.shuffle(items)
        if len(items) <= 1:
            return items, []
        n_val = max(1, int(round(len(items) * val_frac)))
        n_val = min(n_val, len(items) - 1)
        return items[n_val:], items[:n_val]

    tr_p, va_p = split_one(pos)
    tr_n, va_n = split_one(neg)
    return tr_p + tr_n, va_p + va_n


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


def bgr_to_model_input(bgr: np.ndarray, size: int, align_preprocess: bool, cfg: dict | None) -> np.ndarray | None:
    import cv2

    if bgr is None or bgr.size == 0:
        return None
    if align_preprocess and cfg is not None:
        root = _repo_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.vision_preprocess import apply_vision_preprocess

        bgr = apply_vision_preprocess(bgr, cfg)
        if bgr is None or bgr.size == 0:
            return None
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    x = rgb.astype(np.float32) / 127.5 - 1.0
    return x


def batch_tensors(
    items: list[tuple[Path, int]],
    size: int,
    align_preprocess: bool,
    cfg: dict | None,
    augment_flip: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for p, y in items:
        bgr = load_bgr(p)
        if bgr is None:
            continue
        if augment_flip and rng.random() < 0.5:
            bgr = np.ascontiguousarray(bgr[:, ::-1, :])
        t = bgr_to_model_input(bgr, size, align_preprocess, cfg)
        if t is None:
            continue
        xs.append(t)
        ys.append(float(y))
    if not xs:
        return None
    return np.stack(xs), np.array(ys, dtype=np.float32)


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(np.int32)
    y = labels.astype(np.int32)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "threshold": thr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision_shoe": round(prec, 4),
        "recall_shoe": round(rec, 4),
        "f1_shoe": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def best_threshold_f1(probs: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    best_thr, best_m = 0.5, metrics_at_threshold(probs, labels, 0.5)
    best_f1 = best_m["f1_shoe"]
    for thr in [x * 0.02 for x in range(5, 48)]:
        m = metrics_at_threshold(probs, labels, thr)
        if m["f1_shoe"] >= best_f1:
            best_f1 = m["f1_shoe"]
            best_thr = thr
            best_m = m
    return best_thr, best_m


def main() -> None:
    parser = argparse.ArgumentParser(
        epilog="Tip: put hard negatives in datasets/shoe_binary/not_shoe and extra shoes in datasets/shoe_binary/shoe.",
    )
    parser.add_argument(
        "--pos",
        nargs="+",
        default=[
            "datasets/shoe_binary/shoe",
            "datasets/shoes",
            "datasets/dirty_Shoe",
        ],
        help="Positive (shoe) image roots",
    )
    parser.add_argument(
        "--neg",
        nargs="+",
        default=[
            "datasets/shoe_binary/not_shoe",
            "datasets/not_shoe",
        ],
        help="Negative (not shoe) roots",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--finetune_epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--finetune_lr", type=float, default=None, help="default: lr/5")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="models/shoe_binary.tflite")
    parser.add_argument(
        "--align_preprocess",
        action="store_true",
        help="Apply config.yaml vision_preprocess (ROI+CLAHE) — use when images are full booth frames",
    )
    parser.add_argument("--no_augment", action="store_true", help="Disable random horizontal flip")
    args = parser.parse_args()

    base = _repo_root()
    pos_paths: list[tuple[Path, int]] = []
    neg_paths: list[tuple[Path, int]] = []
    for d in args.pos:
        pos_paths.extend(collect((base / d).resolve(), 1))
    for d in args.neg:
        neg_paths.extend(collect((base / d).resolve(), 0))

    pos_paths = merge_unique(pos_paths)
    neg_paths = merge_unique(neg_paths)

    n_pos, n_neg = len(pos_paths), len(neg_paths)
    if n_pos < 4 or n_neg < 4:
        raise SystemExit(
            f"Need at least 4 images per class (got shoe={n_pos}, not_shoe={n_neg}). "
            "Add files under datasets/shoe_binary/shoe, datasets/shoe_binary/not_shoe, "
            "datasets/shoes, datasets/not_shoe, or datasets/dirty_Shoe."
        )

    cfg: dict | None = None
    if args.align_preprocess:
        import yaml

        cfg_path = base / "config.yaml"
        if not cfg_path.is_file():
            raise SystemExit("config.yaml not found for --align_preprocess")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    rng = random.Random(args.seed)
    train_items, val_items = stratified_split(pos_paths, neg_paths, args.val_frac, args.seed)
    if not val_items:
        raise SystemExit("Validation split is empty; add more images or lower --val_frac")

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    def run_epoch(model: keras.Model, items: list, train: bool) -> tuple[float, float]:
        rng.shuffle(items)
        losses, accs = [], []
        for i in range(0, len(items), args.batch):
            chunk = items[i : i + args.batch]
            bt = batch_tensors(
                chunk,
                args.size,
                args.align_preprocess,
                cfg,
                augment_flip=train and not args.no_augment,
                rng=rng,
            )
            if bt is None:
                continue
            xb, yb = bt
            n0 = float(np.sum(yb == 0))
            n1 = float(np.sum(yb == 1))
            sw = np.where(yb == 0, (n0 + n1) / (2.0 * max(n0, 1.0)), (n0 + n1) / (2.0 * max(n1, 1.0)))
            if train:
                h = model.train_on_batch(xb, yb, sample_weight=sw)
            else:
                h = model.test_on_batch(xb, yb, sample_weight=sw)
            losses.append(float(h[0]))
            accs.append(float(h[1]))
        return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(args.size, args.size, 3),
    )
    base_model.trainable = False
    inputs = keras.Input(shape=(args.size, args.size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"train={len(train_items)} val={len(val_items)} shoe={n_pos} not_shoe={n_neg} size={args.size}")
    for epoch in range(args.epochs):
        tr_l, tr_a = run_epoch(model, train_items, train=True)
        va_l, va_a = run_epoch(model, val_items, train=False)
        print(f"epoch {epoch + 1}/{args.epochs} train_loss={tr_l:.4f} acc={tr_a:.4f}  val_loss={va_l:.4f} val_acc={va_a:.4f}")

    base_model.trainable = True
    ft_lr = args.finetune_lr if args.finetune_lr is not None else args.lr / 5
    model.compile(
        optimizer=keras.optimizers.Adam(ft_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    for epoch in range(max(1, args.finetune_epochs)):
        tr_l, tr_a = run_epoch(model, train_items, train=True)
        va_l, va_a = run_epoch(model, val_items, train=False)
        print(
            f"finetune {epoch + 1}/{args.finetune_epochs} train_loss={tr_l:.4f} acc={tr_a:.4f}  "
            f"val_loss={va_l:.4f} val_acc={va_a:.4f}"
        )

    def predict_all(items: list[tuple[Path, int]]) -> tuple[np.ndarray, np.ndarray]:
        probs: list[float] = []
        labels: list[float] = []
        for i in range(0, len(items), args.batch):
            chunk = items[i : i + args.batch]
            bt = batch_tensors(
                chunk,
                args.size,
                args.align_preprocess,
                cfg,
                augment_flip=False,
                rng=rng,
            )
            if bt is None:
                continue
            xb, yb = bt
            pb = model.predict(xb, verbose=0).flatten()
            probs.extend(pb.tolist())
            labels.extend(yb.tolist())
        return np.array(probs, dtype=np.float32), np.array(labels, dtype=np.float32)

    val_probs, val_labels = predict_all(val_items)
    train_probs, train_labels = predict_all(train_items)
    if val_probs.size == 0:
        raise SystemExit("No validation images could be loaded (check paths and formats).")

    thr_default = 0.55
    m_def = metrics_at_threshold(val_probs, val_labels, thr_default)
    thr_best, m_best = best_threshold_f1(val_probs, val_labels)
    m_train_best = metrics_at_threshold(train_probs, train_labels, thr_best)

    print("\n--- Validation metrics ---")
    print(f"threshold={thr_default} (config default-ish): {m_def}")
    print(f"threshold={thr_best:.2f} (max F1 on val): {m_best}")
    print(f"(train at thr={thr_best:.2f}): {m_train_best}")

    out_path = (base / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    print(f"\nWrote {out_path} ({len(tflite_bytes)} bytes)")

    metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
    report = {
        "model": str(out_path.relative_to(base)),
        "input_size": args.size,
        "align_preprocess": args.align_preprocess,
        "counts": {"shoe": n_pos, "not_shoe": n_neg, "train": len(train_items), "val": len(val_items)},
        "val_metrics_default_threshold": m_def,
        "val_metrics_best_f1_threshold": m_best,
        "suggested_config": {
            "shoe_binary.enabled": True,
            "shoe_binary.threshold": round(float(thr_best), 2),
            "shoe_binary.input_size": args.size,
        },
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print("\nEnable on Pi: set shoe_binary.enabled: true in config.yaml (use suggested threshold from JSON).")


if __name__ == "__main__":
    main()
