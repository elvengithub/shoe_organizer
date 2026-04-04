# Shoe type reference images (sports · casual · leather)

Add **several photos per type** taken from your cleaning bay (same camera angle and crop you use live). Use `.jpg`, `.png`, or `.webp`.

```
datasets/shoe_types/
  sports/    ← running shoes, trainers, cleats, etc.
  casual/    ← everyday sneakers, sandals, canvas, etc.
  leather/   ← dress shoes, loafers, polished leather boots, etc.
```

The server compares the live frame to these folders with the same histogram matcher as `datasets/shoes`. Tune `shoe_type_dataset` in `config.yaml` (`min_match_score`, `min_margin`) if the wrong type wins.

If a folder is **empty**, that type is ignored. If **no folder** has images, classification falls back to catalog folders + OpenCV heuristics only.
