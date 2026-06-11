# Dark-Vessel Detection on SAR

**Finding the ships that don't want to be found.**

Roughly 1 in 5 large commercial fishing vessels operates "dark" — present on
radar but broadcasting no AIS identity — and a disproportionate share of that
activity happens inside protected waters. Satellite radar (SAR) sees through
cloud and darkness and doesn't care whether a ship is broadcasting.

This project trains a detector on **xView3-SAR** (~1,000 Sentinel-1 scenes,
220k+ labelled vessels) that doesn't just box the ships — it separates
**dark vs. cooperative** vessels via SAR × AIS data fusion and turns the
detections into a map of **estimated illegal-fishing pressure inside a marine
protected area**, with uncertainty bounds.

> The radar sees it. AIS doesn't. **That discrepancy is the signal.**

An honest technical note: "dark" is not a visual class — a dark trawler and a
cooperative trawler scatter microwaves identically. Darkness is the *result of
a fusion step*: a SAR detection that fails to correlate with any AIS message
in a space–time window. The ML contribution is where ML is genuinely needed:
high-recall detection of tiny point targets, vessel/fishing classification,
and length estimation.

## Pipeline

| Stage | What | Where |
|---|---|---|
| 1 | Tiling: windowed reads, overlap, dB normalisation, land masking, geo-referencing | `src/dark_vessel/data/tiling.py` |
| 2 | Center-heatmap detector (U-Net + ResNet encoder, focal loss) | `src/dark_vessel/models/` |
| 3 | Attribute heads: vessel / fishing / length | `src/dark_vessel/models/model.py` |
| 4 | SAR × AIS correlation → dark / cooperative | `src/dark_vessel/fusion/ais_correlation.py` |
| 5 | WDPA join → MPA pressure map + dark-vessel-hours estimate with 95% CI | `src/dark_vessel/analysis/`, `visualization/` |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify the whole pipeline on synthetic data (no dataset needed):
python scripts/00_smoke_test.py
```

Then follow **[INSTRUCTIONS.txt](INSTRUCTIONS.txt)** step by step — it covers
dataset registration/download (xView3, WDPA), tiling, training, inference,
the dark split, and the final map.

## Documentation

- **[INSTRUCTIONS.txt](INSTRUCTIONS.txt)** — every step you must do, in order.
- **[PROJECT_EXPLAINED.txt](PROJECT_EXPLAINED.txt)** — what the project is and how each stage works.
- **[LEARN_THE_CONCEPTS.txt](LEARN_THE_CONCEPTS.txt)** — a deep-dive teaching guide: SAR physics, geospatial engineering, heatmap detection, focal loss, data fusion, bootstrap uncertainty, mission-aligned evaluation.

## Evaluation

More than a single F1 (see `evaluation/metrics.py`): detection F1 within
200 m, **recall on the dark subset** (the metric that actually matters),
close-to-shore recall, vessel/non-vessel and fishing F1, length score, and an
xView3-style aggregate.

## Data & licences

xView3-SAR is free after registration; WDPA polygons from Protected Planet.
Verify current dataset terms and AIS data-use conditions before publishing
results; treat all effort estimates as uncertainty-bounded extrapolations.
