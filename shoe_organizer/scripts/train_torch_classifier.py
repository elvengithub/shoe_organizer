#!/usr/bin/env python3
"""
Train one classifier head (shoe type OR cleanliness) with transfer learning,
class-balanced sampling, and Albumentations (brightness, rotation, blur, CLAHE).

Expected layout::

  data_root/
    clean/   *.jpg
    dirty/   *.jpg

Or for shoe type::

  data_root/
    sneaker/
    boot/
    ...

Run from the ``shoe_organizer`` app directory::

  python scripts/train_torch_classifier.py --data ../dataset/clean --out models/cleanliness_classifier.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.preprocess import build_train_augmentation  # noqa: E402


def _fixed_eval_pipeline(input_size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _build_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models

    name = backbone.lower().strip()
    if name in ("mobilenet_v3_small", "mobilenet"):
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if name in ("resnet18", "resnet"):
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise SystemExit(f"Unknown backbone {backbone!r} — use mobilenet_v3_small or resnet18")


def _collect_items(data_root: Path, class_names: list[str]) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for i, name in enumerate(class_names):
        d = data_root / name
        if not d.is_dir():
            raise SystemExit(f"Missing class folder: {d}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                paths.append(p)
                labels.append(i)
    if not paths:
        raise SystemExit(f"No images found under {data_root}")
    return paths, labels


class AlbumentationsDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        arr = np.array(img)
        out = self.transform(image=arr)
        return out["image"], int(self.labels[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names (label index order). Default: subfolders sorted by name.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.classes:
        class_names = [x.strip() for x in args.classes.split(",") if x.strip()]
        if len(class_names) < 2:
            raise SystemExit("--classes needs at least two names")
    else:
        order_path = args.data / "class_order.txt"
        if order_path.is_file():
            ordered: list[str] = []
            for line in order_path.read_text(encoding="utf-8").splitlines():
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                if (args.data / name).is_dir():
                    ordered.append(name)
            class_names = ordered if len(ordered) >= 2 else []
        else:
            class_names = []
        if len(class_names) < 2:
            class_names = sorted([p.name for p in args.data.iterdir() if p.is_dir()])
        if len(class_names) < 2:
            raise SystemExit("Need at least two class subfolders under --data (or class_order.txt)")

    paths, labels = _collect_items(args.data, class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idxs = list(range(len(paths)))
    random.shuffle(idxs)
    n_val = max(1, int(0.15 * len(idxs)))
    val_idx_set = set(idxs[:n_val])
    train_indices = [i for i in range(len(paths)) if i not in val_idx_set]
    val_indices = [i for i in range(len(paths)) if i in val_idx_set]

    train_paths = [paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_aug = build_train_augmentation(args.input_size, normalize_lighting=True)
    eval_aug = _fixed_eval_pipeline(args.input_size)

    train_ds = AlbumentationsDataset(train_paths, train_labels, train_aug)
    val_ds = AlbumentationsDataset(val_paths, val_labels, eval_aug)

    counts = Counter(train_labels)
    sample_w = [1.0 / max(counts[y], 1) for y in train_labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = _build_model(args.backbone, len(class_names), args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
        train_acc = correct / max(total, 1)

        model.eval()
        vt, vc = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                vc += int((pred == y).sum().item())
                vt += y.size(0)
        val_acc = vc / max(vt, 1)
        print(
            f"epoch {epoch + 1}/{args.epochs} train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"loss={loss_sum / max(total, 1):.4f}"
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "state_dict": state,
        "class_names": class_names,
        "backbone": args.backbone,
        "input_size": args.input_size,
        "val_acc": float(best_acc),
    }
    torch.save(payload, args.out)
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
