# Project checkpoint — **4:04 PM**

Use this label in chat when you want to refer back to **this recorded state** of the shoe organizer.

## What this version includes

- **Run layout:** Outer folder `run.py` + `requirements.txt` forward to inner `shoe_organizer/`; `pyserial` in dependencies; optional `serial` import handling in `serial_bridge.py`.
- **README:** Short start + dependencies only.
- **UI:** `bay-layout` — camera frame sized to stream (640×480), storage slots 2–6 beside camera in a 3×2 grid (2/3, 4/5, 6), height leveled to preview; CSS `flex-wrap: nowrap` on wide screens, responsive breakpoint ~1100px.
- **Classification / API:** Not-shoe pipeline with stable `classification_error` codes, `reject_detail`, `pipeline_error` / `analysis_failed` path; intake + overlay handling in `index.html`.
- **Shoe types (three only):** **Sports**, **Casual**, **Leather** — vision heuristics in `vision_service.py` (saturation, edges, Sobel gradient); `shoe_taxonomy.py` maps catalog + vision to those three; `wash_decision.py` and text path aligned.
- **Type dataset:** `datasets/shoe_types/{sports,casual,leather}/` + `shoe_type_dataset.py` histogram matcher + `config.yaml` block `shoe_type_dataset`; wired in `ai_camera.py` with `shoe_type_dataset_*` API fields.

## Paths (inner project root) 

- Code: `src/` (app, orchestrator, ai_camera, shoe_type_dataset, vision, etc.)
- Config: `config.yaml`
- Front end: `templates/index.html`, `static/style.css`
- Datasets: `datasets/shoe_types/`, `datasets/shoes/` (if present), `datasets/not_shoe/` (if present)

## Note

This file is a **human-readable snapshot** only, not an automatic backup. For a real restore point, use **git commit/tag** with message e.g. `checkpoint 4:04pm`.
