# Shoe type reference images (sports · casual · leather)

Add **several photos per type** taken from your cleaning bay (same camera angle and crop you use live). Use `.jpg`, `.png`, or `.webp`.

```
datasets/shoe_types/
  sports/    ← running shoes, trainers, cleats, etc.
  casual/    ← everyday sneakers, sandals, canvas, etc.
  leather/   ← dress shoes, loafers, polished leather boots, etc.
  sports/clean/  ← optional: clean booth photos (preferred as refs when present)
  casual/clean/
```

The Flask app loads paths relative to the **inner** project folder `shoe_organizer/` (see `shoe_organizer/config.yaml` → `shoe_type_dataset.path`). The same `datasets/shoe_types/` tree also exists at the **repository root** under `datasets/shoe_types/` so images are visible on GitHub from the top level of the repo; keep both copies in sync when you add new reference photos.

The server compares the live frame to these folders with the same histogram matcher as `datasets/shoes`, then **fuses** those scores with live OpenCV cues and your `datasets/shoes` folder names (see `shoe_type_dataset.fusion` in `config.yaml`). Use **`top_k_refs`** (mean of best K matches per type) for stabler results. Tune `weight_histogram`, `weight_vision`, `weight_catalog`, and `temperature` if the wrong type wins.

If a folder is **empty**, that type is ignored. If **no folder** has images, classification falls back to catalog folders + OpenCV heuristics only.
