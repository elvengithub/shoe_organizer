"""
Derive training / UI type labels from ``dataset/type/<class>/`` layout (folder names = classes).

When ``ai_pipeline.dataset.sync_type_labels_from_dir`` is true, ``load_config`` fills
``type_class_names`` and missing ``type_to_bucket`` entries so the camera pipeline
stays aligned with how you organize images on disk.
"""
from __future__ import annotations

from pathlib import Path

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _has_any_image(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXT:
            return True
    return False


def read_type_class_names(
    type_dir: Path,
    *,
    class_order_file: str | None = "class_order.txt",
) -> list[str]:
    """
    Class names = subfolders of ``type_dir`` that contain at least one image.

    If ``class_order_file`` exists inside ``type_dir``, its non-empty lines define
    order and membership (folders listed must exist). Otherwise folders are sorted
    alphabetically (stable, reproducible).
    """
    if not type_dir.is_dir():
        return []

    if class_order_file:
        order_path = type_dir / class_order_file
        if order_path.is_file():
            names: list[str] = []
            for line in order_path.read_text(encoding="utf-8").splitlines():
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                sub = type_dir / name
                if sub.is_dir() and _has_any_image(sub):
                    names.append(name)
            return names

    names = []
    for p in sorted(type_dir.iterdir()):
        if p.is_dir() and _has_any_image(p):
            names.append(p.name)
    return names


def apply_dataset_derived_type_config(cfg: dict, app_root: Path | None = None) -> None:
    """Mutates ``cfg`` in place when sync is enabled."""
    ap = cfg.get("ai_pipeline")
    if not isinstance(ap, dict):
        return
    ds = ap.get("dataset")
    if not isinstance(ds, dict) or not bool(ds.get("sync_type_labels_from_dir", False)):
        return

    root = app_root or Path(__file__).resolve().parent.parent
    rel = str(ds.get("type_dir", "dataset/type"))
    type_dir = (root / rel).resolve()
    use_order = bool(ds.get("use_class_order_file", True))
    cof = ds.get("class_order_file")
    order_fn = (str(cof) if cof else "class_order.txt") if use_order else None
    names = read_type_class_names(type_dir, class_order_file=order_fn)
    if len(names) < 2:
        return

    ap["type_class_names"] = names
    default_bucket = str(ap.get("type_to_bucket_default") or "casual")
    if default_bucket not in ("sports", "casual"):
        default_bucket = "casual"

    tb = dict(ap.get("type_to_bucket") or {})
    for n in names:
        if n not in tb:
            tb[n] = default_bucket
    ap["type_to_bucket"] = tb
    cfg["ai_pipeline"] = ap
