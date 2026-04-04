from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: Path | None = None) -> dict:
    base = Path(__file__).resolve().parent.parent
    p = path or base / "config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_mock_hardware() -> bool:
    return os.environ.get("MOCK_HARDWARE", "").lower() in ("1", "true", "yes")
