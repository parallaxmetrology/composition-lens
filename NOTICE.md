# Notice: data, models and comparators

- **PICD** (Zhao et al., CVPR 2025) is released under CC BY-NC-SA 4.0 with its own Terms of Use (non-commercial academic and educational use; its images come from open platforms and redistributable datasets and the photographer or platform is not recorded per image). This repository contains no PICD image; it contains image ids and labels (public in the PICD repository) and numbers computed from the images. The edition of the note on parallaxmetrology.com shows a few PICD images, reduced and uncropped, with credit and ids.
- **CADB** (Zhang, Niu and Zhang, BMVC 2021) is built on AADB (Flickr images, research use only). No CADB image or annotation is redistributed; `derived/cadb_features.json` contains numbers computed from the images.
- **DINOv3** weights (timm `vit_large_patch16_dinov3.lvd1689m`) are used under their own licence; no weights or embeddings are included.
- **Facet** (github.com/ncoevoet/facet): `lens/facet_style.py` reimplements, from its public description and source reading, a thirds and centre heuristic as a comparator. It is not Facet's code.
- The template vocabulary comes from an overlay list published by Grid Maker Pro; only the names of templates are used.
