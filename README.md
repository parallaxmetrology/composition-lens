# Where the line sits

Code, derived numbers and documents for the note *Where the line sits: reading horizon and vertical-line position in photographs, checked against human marks* (Russell Parrish, Parallax Metrology, 20 November 2026). Read the note: [parallaxmetrology.com/composition-lens](https://parallaxmetrology.com/composition-lens/) or [`note/index.html`](note/index.html); the full lab book is [`note/labbook.html`](note/labbook.html).

## The finding

Photographers describe composition by where things sit: a horizon in the middle or on a third of the frame, a vertical line at the centre or on a third. We built plain, named measurements of the strongest horizontal and vertical lines in a photograph (no model trained on photographs) and checked them two ways, against two public datasets, **PICD** (Zhao et al., CVPR 2025; 24 expert-labelled composition categories) and **CADB** (Zhang, Niu and Zhang, BMVC 2021; 13 composition classes with the annotators' own drawn marks).

| Check | Result |
|---|---|
| CADB: horizon reading against the annotators' marked horizontal line (images with one mark, n = 2,117) | median error 0.040 of the image height, against 0.226 when the marks are shuffled across images; 53% within 0.05 |
| CADB: vertical reading against the marked vertical line (n = 377) | median error 0.069, against 0.287 shuffled; 47% within 0.05 |
| CADB: zero-shot class ranking (test split, AUC) | horizontal 0.77, vertical 0.63 |

PICD pairs that every whole-image method we tried (including three learned models) left near coin-flip, separated by the line readings (AUC; 0.5 is no separation; DevA is the design set, DevB the confirmation set, LockboxB 960 fresh images, each read once):

| Pair | DevA | DevB [95% interval] | LockboxB [95% interval] | Pooled DINOv3 probe, DevB |
|---|---:|---:|---:|---:|
| LS-Hori3 vs LS-Hori2 | 0.85 | 0.86 [0.79, 0.92] | 0.78 [0.67, 0.88] | 0.54 |
| L-Ver3 vs L-Ver2 | 0.77 | 0.82 [0.74, 0.89] | 0.79 [0.68, 0.90] | 0.64 |

![The two PICD line pairs](docs/figures/pair-auc-line-pairs.svg)

![Agreement with the annotators' marked lines on CADB](docs/figures/cadb-line-agreement.svg)

**Not established:** subject placement. A compact-object reading separated a third PICD pair (subject at the centre vs at a power point) but did not follow the annotators' subject boxes on CADB, so that pair is not claimed. Thirds-point and centre-point scores are barely above chance on CADB (AUC 0.57 and 0.55); diagonal, golden-ratio, golden-triangle, spiral and perspective scores are not supported on real photographs (they were built and tested on exact synthetic layouts). A fitted DINOv3 probe is ahead of every zero-shot score on every CADB class, and on PICD learned probes with position-aware features match the line readings given labelled images, so the readings' advantage is that they need none. See the note and lab book for the figures, the comparators and the limits.

## Check it

```
pip install -r requirements.txt
python reproduce.py      # recomputes the PICD numbers (and the CADB numbers if CADB is unzipped into data/cadb/) from derived/
python check_hashes.py   # every file matches SHA256SUMS
```

`reproduce.py` needs no images. It recomputes, to 1e-9, the crosswalk AUCs and bootstrap intervals, the three position pairs and the Facet-style comparator on DevA, DevB and LockboxB from `derived/picd_features_*.json`, and (with CADB present) the CADB class AUCs and the agreement with the marks from `derived/cadb_features.json`. It does not recompute the fitted probes (they need DINOv3 embeddings), the learning curves, or the synthetic tests; the commands for those are in the lab book (section 10).

## Layout

| Path | What it is |
|---|---|
| `note/index.html`, `note/labbook.html` | The note and the lab book (this edition shows no dataset images) |
| `docs/figures/` | The figures above as standalone SVG |
| `lens/` | The readings: `engine.py` (position profiles, template fits against a null), `m2b.py` (line-versus-compact reading, rabatment, golden triangle and spiral, vanishing points), `features.py`, `facet_style.py` (comparator), `synth.py` and `synth2.py` (exact synthetic layouts) |
| `catalogue/` | 23 named composition templates as data, with a draft crosswalk to PICD and CADB categories |
| `tests/` | Every stage as run: synthetic recovery, M2b, PICD (M3), baselines (M3b), the fresh PICD test (M3c), CADB (M4) |
| `plans/` | `PLAN.md` (definitions, predictions, amendments, result log) and the plan snapshots frozen before the confirmation, fresh and CADB runs |
| `derived/` | Numbers computed from the images (readings per image, with PICD labels); no images, no CADB annotations |
| `results/` | Summary outputs of every analysis and the claims ledger; `provenance/` records the frozen specification hashes |
| `labbook/` | The generators for the note and the lab book |
| `data/` | Small PICD manifests and instructions for getting the datasets |

## What is not here, and how to get it

No dataset image, CADB annotation or embedding is redistributed; see `data/README.md` and `NOTICE.md`. To rerun a stage from images, obtain PICD and CADB, set `PICD_ROOT`, freeze the stage specification with `python tools/freeze.py m3` (then `m3c`, `m4`), and run the scripts in the order given in the lab book. The frozen hashes recorded in the working copy are in `results/provenance/`; this copy differs from it only in file paths.

## Provenance

Every stage was written down in `plans/PLAN.md` before it ran, with its predictions and sanity gates. The PICD readings were designed on one half of a held-out pool (DevA), the specification was hashed, the other half (DevB) was read once, and 960 fresh images (LockboxB) were read once under a second hashed specification. CADB was read once, zero-shot, under a third. Changes made after a result was seen are dated amendments in `plans/PLAN.md` (A1 to A6), among them a change to the extraction script after the first CADB extraction hung with no output produced. Several pre-stated predictions were not met and are reported (S-RoT, S-Cent, S-Per on PICD; diagonal, golden, spiral and perspective on CADB; part of the baseline predictions on the horizon pair). The analysis code was written with AI coding assistants (Anthropic Claude).

## Requirements

Python 3.10+, numpy, scipy, scikit-learn, opencv-python, pillow (see `requirements.txt`); torch and timm only for the DINOv3 stages.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). Note, lab book, derived numbers and documentation: CC BY 4.0 ([`LICENSE-CONTENT`](LICENSE-CONTENT)). Datasets keep their own terms ([`NOTICE.md`](NOTICE.md)). To cite, see [`CITATION.cff`](CITATION.cff).
