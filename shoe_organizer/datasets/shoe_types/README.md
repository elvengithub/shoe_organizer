# Shoe type reference images (sports · casual)

Add **several photos per type** from your cleaning bay (same camera angle and crop as the live view). Use `.jpg`, `.png`, or `.webp`.

```
datasets/shoe_types/
  sports/    ← trainers, runners, mesh athletic shoes you want labeled “sports”
  casual/    ← smooth sneakers, canvas, minimal slip-ons you want labeled “casual”
  sports/clean/  ← clean booth photos (used as refs when present; see `reference_subfolder` in config)
  casual/clean/
```

Image files are **not** tracked in git (see root `.gitignore`); add them locally after clone.

With `vision.rule_based_pipeline: true` and `shoe_type_dataset.refine_rule_based_type: true` in `config.yaml`, the server fuses OpenCV cues with histogram matches to your reference folders.

If a folder is **empty**, that type is ignored for matching.

A legacy `leather/` folder, if present, is treated as extra **casual** references.
