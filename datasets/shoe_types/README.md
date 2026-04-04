# Shoe type reference images (sports · casual · leather)

Add **several photos per type** taken from your cleaning bay (same camera angle and crop you use live). Use `.jpg`, `.png`, or `.webp`.

```
datasets/shoe_types/
  sports/    ← running shoes, trainers, cleats, etc.
  casual/    ← everyday sneakers, sandals, canvas, etc.
  leather/   ← dress shoes, loafers, polished leather boots, etc.
```

The server compares the live frame to these folders with the same histogram matcher as `datasets/shoes`, then **fuses** those scores with live OpenCV cues and your `datasets/shoes` folder names (see `shoe_type_dataset.fusion` in `config.yaml`). Use **`top_k_refs`** (mean of best K matches per type) for stabler results. Tune `weight_histogram`, `weight_vision`, `weight_catalog`, and `temperature` if the wrong type wins.

If a folder is **empty**, that type is ignored. If **no folder** has images, classification falls back to catalog folders + OpenCV heuristics only.
