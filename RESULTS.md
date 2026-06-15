# Results & methods

Everything here is measured on xView3 validation scenes — scenes the model
never saw while training. The model trained on 29 Sentinel-1 scenes; it's a
center-heatmap U-Net with a ResNet-34 encoder, working on 1024×1024 chips.
If you want the reasoning behind the design choices, that's in
[LEARN_THE_CONCEPTS.txt](LEARN_THE_CONCEPTS.txt).

## 1. Detection and attribute metrics

Three Adriatic validation scenes, with a detection counted correct if it lands
within 200 m of a real vessel:

| Scene | Region | Det. F1 | Det. P | Det. R | Dark recall | Near-shore R | Fishing F1 |
|---|---|---|---|---|---|---|---|
| `0d8ed29b…` | Ravenna / Kornati | 0.670 | 0.838 | 0.558 | 0.298 (94 dark) | 0.082 | 0.958 |
| `13dd786e…` | Ravenna / Po delta | 0.685 | 0.905 | 0.551 | 0.336 (107 dark) | 0.067 | 0.903 |
| `3808f570…` | Isole Tremiti | 0.571 | 0.783 | 0.450 | 0.222 (63 dark) | 0.154 | 0.941 |

A few things worth pointing out:

- Precision is high across the board (0.78–0.91), so when the model flags
  something as a vessel, it's usually right. It errs on the side of caution,
  which is the behaviour you want if the output is going to a patrol that has
  to decide where to look.
- Dark recall (around 0.22–0.34) is the weak spot, and it's the number that
  matters most here. Overall detection recall is ~0.55, but on the dark subset
  it falls to ~0.30 — the model finds vessels in general but misses a lot of
  the small, low-signature dark fishing boats. That's not a coincidence; dark
  vessels tend to *be* the hard targets.
- Near-shore recall is low (0.07–0.15). Coastline clutter — rocks, surf, fixed
  structures — is genuinely the hardest case, and this is where the model
  struggles most. Better to say so than to hide it.

## 2. Did more data help? (baseline vs final)

The baseline trained on 5 scenes, the final model on 29.

| Scene | Det. F1 (base → final) | Dark recall (base → final) |
|---|---|---|
| `0d8ed29b…` | 0.679 → 0.670 | 0.330 → 0.298 |
| `13dd786e…` | 0.682 → 0.685 | 0.327 → 0.336 |
| `3808f570…` | 0.503 → 0.571 | 0.159 → 0.222 |

The extra data mostly helped the scene that was struggling — Tremiti gained
about 0.07 on F1 and roughly 40% on dark recall. The two scenes that were
already doing okay stayed flat. That's the usual shape of a data-scaling
curve: you lift the floor before you lift the ceiling. At 29 scenes the model
has basically converged; squeezing out more would take a lot more data or
techniques aimed specifically at tiny targets (hard-negative mining, finer
crops), which I've left as future work.

## 3. The dark / cooperative split

Each detection inherits its AIS-correlation status from the xView3 ground
truth (HIGH confidence means it matched an AIS track, so it's cooperative;
MEDIUM means it was visible in radar but had no AIS match, so it's dark).
Scene-wide counts:

| Scene | Total detections | Dark | Cooperative |
|---|---|---|---|
| `0d8ed29b…` | 185 | 27 | 61 |
| `13dd786e…` | 137 | 34 | 28 |
| `3808f570…` | 69 | 14 | 30 |

## 4. Dark vessels inside protected areas

Detections joined to the WDPA marine-protected-area polygons (Italy and
Croatia).

**Isole Tremiti** (`3808f570…`) — the clean, recognisable one:

| MPA | Total | Dark | Cooperative | Dark ratio |
|---|---|---|---|---|
| Isole Tremiti | 9 | 5 | 3 | 0.63 |
| Pučinski otoci (HR) | 1 | 1 | 0 | 1.00 |

**Po delta** (`13dd786e…`) — the most dark activity:

| MPA | Total | Dark | Cooperative | Dark ratio |
|---|---|---|---|---|
| Sacca di Goro / Po di Goro / Valle Dindona | 4 | 4 | 0 | 1.00 |
| Adriatico settentrionale – Emilia-Romagna | 2 | 1 | 1 | 0.50 |
| Valle di Gorino | 1 | 1 | 0 | 1.00 |

Worth a note: the busiest scene by raw count is `0d8ed29b…` with 185 ships,
but most of that is cooperative shipping traffic, and only one dark vessel
falls inside any protected area (none inside Kornati National Park). So a
crowded map isn't the same as a meaningful one — what counts is dark vessels
inside the boundaries, not the total number of dots.

## 5. Turning a snapshot into hours

A SAR pass is a single instant. Going from "we saw k dark vessels in this
frame" to "this many dark-vessel-hours over a window" means extrapolating, so
it's worth being explicit about how.

The idea: treat the per-pass count as a sample of how many dark vessels are
present at any given moment (call it λ), and over a window of *T* hours the
expected dark-vessel-hours is roughly λ × T. The count gets divided by the
measured dark recall, since we know we only catch a fraction of the dark
vessels actually there. The uncertainty is carried through with a
bootstrap (or a Gamma posterior when the counts are tiny) and reported as a
95% range. The code for all this is in `src/dark_vessel/analysis/effort.py`.

| MPA | Dark detected | Dark recall used | Window | Dark-vessel-hours (95% range) |
|---|---|---|---|---|
| Isole Tremiti | 5 | 0.222 | 24 h | 541 (351 – 806) |
| Sacca di Goro (Po) | 4 | 0.336 | 24 h | 286 (156 – 490) |

The assumptions, laid out plainly:

1. Dark-vessel presence is roughly steady over the window.
2. Each SAR pass is a fair, unbiased snapshot.
3. Counts are scaled up by the measured dark recall — an explicit correction
   you can check, not a fudge factor.
4. "Hours" means dark-vessel *presence* hours. It's good evidence of fishing
   pressure, but it's not a courtroom-grade measure of actual fishing time.
5. One pass is a single-snapshot estimate, so the range is deliberately wide.
   More passes over the same area would narrow it (future work).

In other words, these are extrapolations with error bars, not measurements —
the same spirit as Global Fishing Watch's "apparent fishing effort."

## 6. Reproducing this

```bash
python scripts/04_predict_scene.py --split validation --scene 3808f5703f0920bfv
python scripts/05_dark_split_and_eval.py --split validation
python scripts/06_mpa_pressure_map.py \
    --detections outputs/predictions/3808f5703f0920bfv_detections_dark.csv \
    --window-hours 24 --dark-recall 0.222 --out outputs/maps/map_tremiti.html
```

The full machine-readable reports are in
`outputs/predictions/evaluation_validation.json`.
