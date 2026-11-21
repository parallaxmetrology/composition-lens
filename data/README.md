# Data

No dataset images, annotations or embeddings are redistributed here.

- **PICD** (Zhao et al., CVPR 2025; CC BY-NC-SA 4.0 and its Terms of Use): https://github.com/CV-xueba/PICD_ImageComposition . `data/picd/results/split.json` lists the 3,312 pool image ids with their category labels (1 to 24) and the DevA (0) / DevB (1) split; `data/picd/data/pick_lockboxb.json` lists the 960 LockboxB ids, labels and the Drive file ids. To rerun the extraction from images, place the images at `data/picd/data/images/<id>` (pool) and `data/picd/data/lockboxb/<id>`, or set `PICD_ROOT` to a folder with that layout.
- **CADB** (Zhang, Niu and Zhang, BMVC 2021; AADB Flickr images, research use): https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB . Unzip `CADB_Dataset.zip` into `data/cadb/` so that `data/cadb/CADB_Dataset/images/` and `composition_elements.json` exist.
- The fitted DINOv3 comparators use embeddings from the original environment (not included): `results/emb_dinov3_web_l16.npy` (3,312 x 1,024, pool order of `split.json`), `results/fitted_dino_lockboxb.npy` (960 x 1,024, pick order) under `PICD_ROOT`, and `tests/m3b_extract_dino.py`, `tests/m3c_extract_dino.py`, `tests/m4_extract_dino.py` for the patch-grid and CADB features.
