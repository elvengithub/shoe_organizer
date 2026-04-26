from __future__ import annotations

from pathlib import Path


def app_root() -> Path:
    """Directory containing config.yaml (the `shoe_organizer` package folder)."""
    return Path(__file__).resolve().parent.parent


def resolve_under_app(path_str: str | None, base: Path | None = None) -> Path | None:
    if not path_str:
        return None
    base = base or app_root()
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()
