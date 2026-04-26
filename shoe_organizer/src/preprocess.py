"""
Image preprocessing for classifier inference (resize + ImageNet normalization) and
Albumentations presets for transfer-learning training.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_bgr(bgr: np.ndarray, size: int) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def normalize_lighting_bgr(bgr: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """LAB CLAHE on L channel — matches booth lighting normalization idea."""
    if bgr is None or bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, ch_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(max(2, int(tile)), max(2, int(tile))))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, ch_b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def bgr_to_classifier_batch(bgr: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
    """
    BGR uint8 → float32 NCHW on device, ImageNet normalized.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    t = (t.to(device) - mean) / std
    return t


def build_train_augmentation(
    input_size: int,
    *,
    normalize_lighting: bool = True,
) -> Any:
    """
    Albumentations pipeline: brightness, rotation, blur, optional CLAHE, ImageNet normalize.
    Returns a Compose; use with cv2 BGR images (Albumentations expects RGB — convert in dataset).
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transforms: list[Any] = []
    if normalize_lighting:
        transforms.append(
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.35),
        )
    transforms.extend(
        [
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.55),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.25,
            ),
            A.Resize(input_size, input_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)
