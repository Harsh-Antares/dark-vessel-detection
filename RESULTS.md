# Results & Methods

All numbers below are measured on **xView3 validation scenes** — scenes the
model never saw during training. Training used 29 Sentinel-1 scenes; the
detector is a center-heatmap U-Net (ResNet-34 encoder) trained on
1024×1024 chips. See [LEARN_THE_CONCEPTS.txt](LEARN_THE_CONCEPTS.txt) for the
why behind every design choice.

## 1. Detection & attribute metrics

Three Adriatic validation scenes, matching radius 200 m:

| Scene | Region | Det. F1 | Det. P | Det. R | **Dark recall** | Near-shore R | Fishing F1 |
|---|---|---|---|---|---|---|---|
| `0d8ed29b…` | Ravenna / Kornati | 0.670 | 0.838 | 0.558 | **0.298** (94 dark) | 0.082 | 0.958 |
| `13dd786e…` | Ravenna / Po delta | 0.685 | 0.905 | 0.551 | **0.336** (107 dark) | 0.067 | 0.903 |
| `3808f570…` | Isole Tremiti | 0.571 | 0.783 | 0.450 | **0.222** (63 dark) | 0.154 | 0.941 |

**Reading these:**
- **Precision is high (0.78–0.91)** — when the model reports a vessel it is
  almost always right. The detector is deliberately conservative, the correct
  bias for an enforcement-triage tool.
- **Dark recall (~0.22–0.34) is the bottleneck.** Detection recall is ~0.55
  overall, but on the *dark* subset it drops to ~0.30: the model finds vessels
  in general but misses most small, low-signature dark fishing boats. Dark
  vessels are physically the hard targets, and this is the single most
  mission-relevant number — one the standard xView3 aggregate metric ignores.
- **Near-shore recall is very low (0.07–0.15).** Coastal clutter (rocks, surf,
  fixed structures) is the hardest regime; this is an honest, known weak spot.

## 2. Did more training data help? (baseline vs final)

The baseline model saw 5 training scenes; the final model saw 29.

| Scene | Det. F1 (base → final) | Dark recall (base → final) |
|---|---|---|
| `0d8ed29b…` | 0.679 → 0.670 | 0.330 → 0.298 |
| `13dd786e…` | 0.682 → 0.685 | 0.327 → 0.336 |
| `3808f570…` | 0.503 → **0.571** | 0.159 → **0.222** |

More data **lifted the hardest scene most** (Tremiti: +0.07 F1, +40% relative
dark recall) and left the already-near-saturated scenes flat — the expected
shape of a data-scaling curve. The model has effectively converged on 29
scenes; further gains would require substantially more data and/or
small-target-specific techniques (hard-negative mining, higher-resolution
crops), which are listed as future work.

## 3. The dark/cooperative split

For each detection we inherit the AIS-correlation status from the xView3
ground truth (confidence HIGH = AIS-matched = cooperative; MEDIUM =
radar-visible, no AIS match = dark). Scene-wide counts:

| Scene | Total detections | Dark | Cooperative |
|---|---|---|---|
| `0d8ed29b…` | 185 | 27 | 61 |
| `13dd786e…` | 137 | **34** | 28 |
| `3808f570…` | 69 | 14 | 30 |

## 4. Dark vessels inside protected areas (the artifacts)

Detections spatially joined to WDPA marine-protected-area polygons
(Italy + Croatia):

**Isole Tremiti** (`3808f570…`) — the recognisable headline:

| MPA | Total | Dark | Cooperative | Dark ratio |
|---|---|---|---|---|
| Isole Tremiti | 9 | **5** | 3 | 0.63 |
| Pučinski otoci (HR) | 1 | 1 | 0 | 1.00 |

**Po delta** (`13dd786e…`) — the highest dark pressure:

| MPA | Total | Dark | Cooperative | Dark ratio |
|---|---|---|---|---|
| Sacca di Goro / Po di Goro / Valle Dindona | 4 | 4 | 0 | 1.00 |
| Adriatico settentrionale – Emilia-Romagna | 2 | 1 | 1 | 0.50 |
| Valle di Gorino | 1 | 1 | 0 | 1.00 |

(For contrast, the busiest scene `0d8ed29b…` — 185 ships — is mostly
*cooperative* shipping traffic, with only 1 dark vessel inside any MPA and
none inside Kornati National Park. Raw ship count is not the signal; dark
vessels inside protected boundaries is.)

## 5. From snapshots to hours — the extrapolation

Each SAR pass is one instantaneous frame. Converting "k dark vessels seen in
a snapshot" into "dark-vessel-hours over a window" is an extrapolation, and we
treat it as one.

**Model.** Per-pass dark counts are treated as samples of an instantaneous
abundance λ (dark vessels present at any moment). Over a window of *T* hours,
expected dark-vessel-hours **H = λ · T**. Counts are corrected by the measured
dark recall (we catch a known fraction of dark vessels, so the true count is
higher). Uncertainty is propagated with a bootstrap / Gamma-posterior and
reported as a 95% interval. (Code: `src/dark_vessel/analysis/effort.py`.)

| MPA | Dark detected | Dark recall used | Window | **Dark-vessel-hours (95% CI)** |
|---|---|---|---|---|
| Isole Tremiti | 5 | 0.222 | 24 h | **541 [351 – 806]** |
| Sacca di Goro (Po) | 4 | 0.336 | 24 h | **286 [156 – 490]** |

**Assumptions (stated, not hidden):**
1. Dark-vessel presence is roughly stationary over the window.
2. Each SAR pass is an unbiased instantaneous sample.
3. Counts are corrected by the *measured* dark recall — an explicit, auditable
   correction, not a fudge factor.
4. "Hours" means dark-vessel-**presence**-hours — strong evidence of
   illegal-fishing pressure, **not** a court-ready measure of fishing time.
5. A single SAR pass is a one-snapshot estimate; the interval is wide by
   design. Multiple passes over the same MPA would tighten it (future work).

This is an **extrapolation, not a measurement** — the framing mirrors Global
Fishing Watch's "apparent fishing effort."

## 6. Reproduce

```bash
python scripts/04_predict_scene.py --split validation --scene 3808f5703f0920bfv
python scripts/05_dark_split_and_eval.py --split validation
python scripts/06_mpa_pressure_map.py \
    --detections outputs/predictions/3808f5703f0920bfv_detections_dark.csv \
    --window-hours 24 --dark-recall 0.222 --out outputs/maps/map_tremiti.html
```

Full machine-readable reports: `outputs/predictions/evaluation_validation.json`.
