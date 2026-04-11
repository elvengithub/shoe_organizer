# Shoe type reference images (sports · casual)

Add **several photos per type** from your cleaning bay (same camera angle and crop as the live view). Use `.jpg`, `.png`, or `.webp`.

```
datasets/shoe_types/
  sports/    ← trainers, runners, mesh athletic shoes you want labeled “sports”
  casual/    ← smooth sneakers, canvas, minimal slip-ons you want labeled “casual”
```

With `vision.rule_based_pipeline: true` and `shoe_type_dataset.refine_rule_based_type: true` in `config.yaml`, the server:

1. Computes **fusion** cues on the live frame (edges, texture, saturation).
2. Compares the frame to your **sports** and **casual** folders (HSV + gray + LAB histograms, same as the main shoe catalog).
3. If the histogram match passes `min_match_score` and `min_margin`, that **overrides** the fusion label for the final sports/casual decision.

If a folder is **empty**, that type is ignored. If the match is not confident enough, the answer stays **fusion-only** (see API `shoe_type_dataset_scores` and `inference_backend`).

A legacy `leather/` folder, if present, is treated as extra **casual** references.
