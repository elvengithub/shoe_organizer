"""
Stage 2: separate PyTorch image classifiers (shoe type vs cleanliness).
Backbones: MobileNetV3-Small (default, low power) or ResNet18.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .preprocess import bgr_to_classifier_batch


def _build_backbone(
    name: str,
    num_classes: int,
    *,
    pretrained: bool,
) -> nn.Module:
    name_l = name.lower().strip()
    if name_l in ("mobilenet_v3_small", "mobilenetv3_small", "mobilenet"):
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if name_l in ("resnet18", "resnet"):
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown backbone {name!r} — use mobilenet_v3_small or resnet18")


def _pick_device(preferred: str) -> torch.device:
    p = (preferred or "auto").lower().strip()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchImageClassifier:
    """Loads weights from ``train_torch_classifier.py`` checkpoints."""

    def __init__(
        self,
        weights: Path,
        class_names: list[str],
        backbone: str,
        *,
        pretrained_backbone: bool = True,
        input_size: int = 224,
        device_pref: str = "auto",
    ) -> None:
        self.class_names = list(class_names)
        self.input_size = int(input_size)
        self.device = _pick_device(device_pref)
        try:
            ckpt = torch.load(str(weights), map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(weights), map_location=self.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if isinstance(ckpt, dict) and ckpt.get("class_names"):
            self.class_names = list(ckpt["class_names"])
        if isinstance(ckpt, dict) and ckpt.get("backbone"):
            backbone = str(ckpt["backbone"])
        n_cls = len(self.class_names)
        self.model = _build_backbone(backbone, n_cls, pretrained=pretrained_backbone)
        if isinstance(state, dict):
            self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, bgr: np.ndarray) -> dict[str, float]:
        batch = bgr_to_classifier_batch(bgr, self.input_size, self.device)
        logits = self.model(batch)
        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return {self.class_names[i]: float(prob[i]) for i in range(len(self.class_names))}

    def predict_top(self, bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        probs = self.predict_proba(bgr)
        top = max(probs, key=probs.get)
        return top, float(probs[top]), probs


class DualHeadClassifiers:
    """Model A (type) + Model B (cleanliness) — separate checkpoints, separate forwards."""

    def __init__(
        self,
        type_weights: Path,
        cleanliness_weights: Path,
        type_class_names: list[str],
        cleanliness_class_names: list[str],
        backbone: str,
        *,
        pretrained_backbone: bool,
        input_size: int,
        device_pref: str,
    ) -> None:
        self.type_clf = TorchImageClassifier(
            type_weights,
            type_class_names,
            backbone,
            pretrained_backbone=pretrained_backbone,
            input_size=input_size,
            device_pref=device_pref,
        )
        self.clean_clf = TorchImageClassifier(
            cleanliness_weights,
            cleanliness_class_names,
            backbone,
            pretrained_backbone=pretrained_backbone,
            input_size=input_size,
            device_pref=device_pref,
        )

    def predict_both(self, crop_bgr: np.ndarray) -> tuple[
        tuple[str, float, dict[str, float]],
        tuple[str, float, dict[str, float]],
    ]:
        t = self.type_clf.predict_top(crop_bgr)
        c = self.clean_clf.predict_top(crop_bgr)
        return t, c
