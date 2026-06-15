# Dark-Vessel Detection on SAR

**Finding the ships that don't want to be found.**

Roughly 1 in 5 large commercial fishing vessels operates "dark" — present on
radar but broadcasting no AIS identity — and a disproportionate share of that
activity happens inside protected waters. Satellite radar (SAR) sees through
cloud and darkness and doesn't care whether a ship is broadcasting.

This project trains a detector on **xView3-SAR** (Sentinel-1 scenes, 220k+
labelled vessels), separates **dark vs. cooperative** vessels via SAR × AIS
data fusion, and turns the detections into a map of **estimated
illegal-fishing pressure inside marine protected areas (MPAs)** of the
Adriatic Sea — with uncertainty bounds.

> The radar sees it. AIS doesn't. **That discrepancy is the signal.**

**The honest technical note (stated up front):** "dark" is *not* a visual
class — a dark trawler and a cooperative trawler scatter microwaves
identically. Darkness is the *result of a data-fusion step*: a SAR detection
that fails to correlate with any AIS message in a space–time window. The
machine learning earns its keep where ML is genuinely needed — high-recall
detection of tiny point targets, vessel/fishing classification, and length
estimation. The dark/cooperative split is data fusion, not pixel-reading.

---

## Headline result

A trained detector, run on Adriatic Sentinel-1 scenes, flags non-broadcasting
("dark") vessels and places them inside named marine protected areas:

| Map | What it shows |
|---|---|
| **Isole Tremiti** (Italy) | **5 dark fishing vessels** detected inside the Isole Tremiti MPA (dark ratio 0.63). Estimated **≈540 dark-vessel-hours** over a 24 h window *(95% CI 350–810)*. |
| **Po delta** (Emilia-Romagna) | **34 dark vessels** scene-wide — the highest dark pressure observed — with **4 inside the Sacca di Goro / Po-delta reserves**. Estimated **≈290 dark-vessel-hours** *(95% CI 160–490)*. |

![Dark vessels inside the Isole Tremiti MPA](assets/map_tremiti.png)
*Red = dark (no AIS). Blue = cooperative (AIS-matched). Green = MPA boundary.
The heat-surface shows dark-vessel density; the banner reports the
dark-vessel-hours estimate with its 95% interval.*

![Dark vessels in the Po-delta reserves](assets/map_ravenna.png)

All effort figures are **uncertainty-bounded extrapolations**, not
measurements — see [RESULTS.md](RESULTS.md) for every assumption.

---

## Evaluation — more than a single F1

Measured on held-out validation scenes the model never saw in training:

| Metric | Baseline (5 train scenes) | Final (29 train scenes) |
|---|---|---|
| Detection F1 (≤200 m) | 0.50 – 0.68 | 0.57 – **0.69** |
| **Dark recall** | 0.16 – 0.33 | **0.22 – 0.34** |
| Detection precision | ~0.95 | 0.78 – 0.95 |
| Fishing-class F1 | — | 0.90 – 0.96 |

**The finding that matters:** detection *recall* is ~0.55 but **dark recall
is ~0.30** — the model reliably finds vessels in general yet misses most
small, faint *dark* fishing boats, which is exactly what dark vessels
physically are. This gap (and the very low near-shore recall) is the
mission-critical number that the standard xView3 aggregate metric never
reports. More training data lifted the hardest scene most (+40% relative dark
recall) and left the near-saturated scenes flat — the expected shape.

Full per-scene breakdown and the baseline-vs-final comparison: [RESULTS.md](RESULTS.md).

---

## Pipeline

| Stage | What | Where |
|---|---|---|
| 1 | Tiling: windowed reads, overlap, dB normalisation, land masking, geo-referencing | `src/dark_vessel/data/tiling.py` |
| 2 | Center-heatmap detector (U-Net + ResNet encoder, focal loss) | `src/dark_vessel/models/` |
| 3 | Attribute heads: vessel / fishing / length | `src/dark_vessel/models/model.py` |
| 4 | SAR × AIS correlation → dark / cooperative | `src/dark_vessel/fusion/ais_correlation.py` |
| 5 | WDPA join → MPA pressure map + dark-vessel-hours with 95% CI | `src/dark_vessel/analysis/`, `visualization/` |

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify the whole pipeline on synthetic data (no dataset needed):
python scripts/00_smoke_test.py
```

Then follow **[INSTRUCTIONS.txt](INSTRUCTIONS.txt)** step by step — dataset
registration/download (xView3, WDPA), tiling, training (locally or on the
Colab GPU notebook in `notebooks/`), inference, the dark split, and the maps.

## Documentation

- **[INSTRUCTIONS.txt](INSTRUCTIONS.txt)** — every step you run, in order.
- **[PROJECT_EXPLAINED.txt](PROJECT_EXPLAINED.txt)** — what the project is and how each stage works.
- **[LEARN_THE_CONCEPTS.txt](LEARN_THE_CONCEPTS.txt)** — deep-dive teaching guide: SAR physics, geospatial engineering, heatmap detection, focal loss, data fusion, bootstrap uncertainty, mission-aligned evaluation.
- **[RESULTS.md](RESULTS.md)** — full numbers, the effort-estimate method, and every assumption.

## Skills demonstrated

Geospatial deep learning · SAR/remote-sensing fundamentals · large-raster
engineering (tiling, windowed I/O, geo-referenced stitching) ·
heatmap/center-point detection · multi-task learning · data fusion (SAR × AIS)
· extreme class imbalance · uncertainty-aware evaluation · turning model
output into a decision-ready geospatial product.

## Data & licences

xView3-SAR is free after registration; WDPA polygons from Protected Planet.
Verify current dataset terms and AIS data-use conditions before publishing
results; treat all effort estimates as uncertainty-bounded extrapolations.
