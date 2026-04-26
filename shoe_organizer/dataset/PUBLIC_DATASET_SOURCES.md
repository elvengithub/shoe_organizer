# Public shoe image sources (browse & download yourself)

Use these to fill **`dataset/type/`** (sneaker, boot, sandal, loafer, other) and **`dataset/cleanliness/`** (clean, dirty). Always read each site’s **license / terms** before use (research-only, commercial, redistribution, etc.).

---

## Free images to browse (download manually)

These are the usual **“free to use”** pools people use for ML. **Still read the license on each site** (Unsplash / Pexels / Pixabay have their own terms; Commons/Openverse show **per-file** CC licenses).

| Where | What to search | Notes |
|--------|----------------|--------|
| **Unsplash** | https://unsplash.com/s/photos/sneakers — also try `boots`, `sandals`, `loafers`, `dirty shoes`, `muddy shoes` | [Unsplash License](https://unsplash.com/license): very permissive for use; follow their rules (e.g. hotlinking if you use the API). |
| **Pexels** | https://www.pexels.com/search/shoes/ | [Pexels license](https://www.pexels.com/license/): free use with a few restrictions; good for collecting training jpgs. |
| **Pixabay** | https://pixabay.com/images/search/shoes/ | [Pixabay license](https://pixabay.com/service/license-summary/): own rules — read before commercial use. |
| **Openverse** | https://openverse.org/search?q=shoe&license_type=commercial,modification | Aggregates **CC** and public-domain works; filter by license in the UI. |
| **Wikimedia Commons** | [Sneakers](https://commons.wikimedia.org/wiki/Category:Sneakers_(footwear)) · [Boots](https://commons.wikimedia.org/wiki/Category:Boots) · [Sandals](https://commons.wikimedia.org/wiki/Category:Sandals) · [Loafers](https://commons.wikimedia.org/wiki/Category:Loafers) | **Each file** has a license tab (CC BY-SA, CC0, etc.). Attribute when required. |
| **Pixnio (CC0)** | https://pixnio.com/objects/shoes-and-clothing/ | Many images marked **CC0** (public domain dedication) — confirm on the image page. |

**Tip:** For **`dataset/cleanliness`**, search things like `muddy shoes`, `dirty sneakers`, `clean white sneakers` on Pexels/Unsplash and save into `dirty/` vs `clean/` yourself. Stock photos are not perfect for your booth light, so mix in **your own** bay photos when you can.

---

## 1. Fine-grained shoe catalog (great for **type** classes)

### UT Zappos50K (UT-Zap50K)

- **Home:** https://vision.cs.utexas.edu/projects/finegrained/utzap50k/
- **What it is:** ~50k catalog-style shoe images; meta-data includes high-level groups (e.g. Shoes / Sandals / Boots / Slippers) and finer types.
- **License / use:** Stated on the page as **academic / non-commercial** for the image zip; cite their papers if you publish.
- **Maps to this project:** Use labels to sort images into `dataset/type/sneaker`, `boot`, `sandal`, `loafer`, and put odd cases in `other` (you’ll need a simple mapping from their taxonomy to your five folders).
- **Download:** Links on the project page (`ut-zap50k-images.zip` or square crop variant). Academic Torrents also mirrors a bundle: https://academictorrents.com/details/3b3cb58f4ccafc6320d06d00f0862a4ba923b510

---

## 2. Kaggle (good for **shoe vs sandal vs boot** style splits)

Check **each dataset’s license** on its Kaggle page before commercial use.

| Dataset | URL | Typical use |
|--------|-----|-------------|
| Shoe vs Sandal vs Boot (~15k) | https://www.kaggle.com/datasets/hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images | Map folders → `sandal`, `boot`, merge “Shoe” into `sneaker` / `loafer` / `other` by hand or rules |
| Footwear 3-class (~3k) | https://www.kaggle.com/datasets/adityakadam1/footwear | Same idea |
| Shoes classification (~13k) | https://www.kaggle.com/datasets/utkarshsaxenadn/shoes-classification-dataset-13k-images | Subfolders often map to brands/types; map into your five classes |
| Sneakers (multi-class) | https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification | Strong source for **`sneaker`** |

**Download:** Install [Kaggle CLI](https://github.com/Kaggle/kaggle-api) or use the site; paths look like `kaggle datasets download -d <owner>/<slug>`.

---

## 3. Object detection & defects (for **YOLO** or weak **clean/dirty** cues)

### Roboflow Universe — shoe defect / damage

- **Example:** https://universe.roboflow.com/shoe-defect/shoe-defect  
- **What it is:** Bounding boxes / defects on footwear (not the same as “mud on shoe,” but useful for **damage vs OK** prototypes).
- **How to use:** Export classification or detection format; for this repo’s **`cleanliness`** head, treat “no defect” crops as cleaner and defect-heavy crops as dirtier **only if** it matches your real booth — ideally you still add **real** clean/dirty bay photos.

### Roboflow — search

- **Search:** https://universe.roboflow.com/search?q=shoe  
- Many projects are **CC BY 4.0** or similar; confirm on each project page.

---

## 4. Clean vs dirty shoes (honest picture)

There is **no** widely standard “clean shoe / dirty shoe” dataset that matches a **cleaning bay** camera. Best practice:

1. **Capture your own:** `dataset/cleanliness/clean/` and `dataset/cleanliness/dirty/` from your booth (same ROI as `vision_preprocess` in `config.yaml`).
2. **Augment dirt:** mud/synthetic overlays sparingly; keep real dirty shoes as the majority so the model matches production.

Optional **proxy** datasets (floor/object clean vs dirty) are **not** shoe-specific — use only as a last resort and expect domain gap.

---

## 5. After you download

Folder layout expected by `scripts/train_torch_classifier.py`:

```text
dataset/type/sneaker/*.jpg
dataset/type/boot/*.jpg
dataset/type/sandal/*.jpg
dataset/type/loafer/*.jpg
dataset/type/other/*.jpg

dataset/cleanliness/clean/*.jpg
dataset/cleanliness/dirty/*.jpg
```

Train (from app root `shoe_organizer/`):

```bash
python scripts/train_torch_classifier.py --data dataset/cleanliness --out models/cleanliness_classifier.pt --classes clean,dirty
python scripts/train_torch_classifier.py --data dataset/type --out models/shoe_type_classifier.pt --classes sneaker,boot,sandal,loafer,other
```

Then restart your app (or `src.two_stage_vision.reset_runtime_cache()` in a REPL) so new weights load.

---

## Quick license reminder

- **Unsplash / Pexels / Pixabay:** use their published license pages; attribution may be appreciated even when not strictly required.
- **Openverse / Wikimedia:** **per-image** license — check before bulk use or redistribution.
- **Zappos50K:** academic / non-commercial for images — read `readme.txt` in the zip.
- **Kaggle / Roboflow:** per-dataset; many allow competition/research; **commercial** use may require a different license.
- Do **not** scrape Google Images or retail sites without permission; use official dataset downloads or your own photos.
