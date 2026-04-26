from __future__ import annotations

import os
from pathlib import Path

import yaml

from .dataset_layout import apply_dataset_derived_type_config


def load_config(path: Path | None = None) -> dict:
    base = Path(__file__).resolve().parent.parent
    p = path or base / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict):
        apply_dataset_derived_type_config(cfg, app_root=base)
    return cfg


def is_mock_hardware() -> bool:
    return os.environ.get("MOCK_HARDWARE", "").lower() in ("1", "true", "yes")
