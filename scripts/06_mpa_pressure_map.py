"""Step 6 — the killer artifact: the MPA illegal-fishing pressure map.

Joins dark/cooperative detections to WDPA marine-protected-area polygons,
estimates dark-vessel-hours with uncertainty bounds, and renders the
interactive HTML map plus a per-MPA summary table and a methods note.

Usage:
    python scripts/06_mpa_pressure_map.py \
        --detections outputs/predictions/SCENE1_detections_dark.csv \
                     outputs/predictions/SCENE2_detections_dark.csv \
        --window-hours 168 --dark-recall 0.85

--window-hours: the time span your SAR passes cover (e.g. 168 = one week).
--dark-recall : the detector's measured recall on the dark subset (from
                step 5's report) — used to correct counts for missed vessels.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from dark_vessel.analysis.effort import estimate_dark_hours
from dark_vessel.analysis.mpa import (join_detections_to_mpas, load_mpas,
                                      mpa_summary)
from dark_vessel.config import load_config
from dark_vessel.visualization.maps import build_pressure_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--detections", nargs="+", required=True,
                    help="one *_detections_dark.csv per SAR pass")
    ap.add_argument("--window-hours", type=float, required=True)
    ap.add_argument("--dark-recall", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="output HTML path")
    args = ap.parse_args()
    cfg = load_config(args.config)

    frames = []
    for path in args.detections:
        df = pd.read_csv(path)
        df["pass_id"] = Path(path).stem
        frames.append(df)
    det = pd.concat(frames, ignore_index=True)
    n_passes = det["pass_id"].nunique()
    print(f"{len(det)} detections across {n_passes} SAR pass(es)")

    # Clip WDPA loading to the detections' bounding box (+ a margin).
    pad = 0.5
    bbox = (det["lon"].min() - pad, det["lat"].min() - pad,
            det["lon"].max() + pad, det["lat"].max() + pad)
    mpas = load_mpas(cfg.paths.wdpa_path, bbox=bbox)
    print(f"{len(mpas)} MPA polygon(s) intersect the detection area")
    if mpas.empty:
        raise SystemExit("No MPAs in this area — pick a scene that covers one.")

    joined = join_detections_to_mpas(det, mpas)
    summary = mpa_summary(joined, mpas, n_passes=n_passes)
    print("\nPer-MPA dark-vessel pressure:")
    print(summary.to_string(index=False))

    maps_dir = Path(cfg.paths.maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(maps_dir / "mpa_summary.csv", index=False)

    # Effort estimate for the MPA with the most dark detections.
    estimate = None
    if not summary.empty and summary.iloc[0]["n_dark"] > 0:
        top = summary.iloc[0]["mpa_name"]
        counts = [
            int(((joined["mpa_name"] == top) & (joined["is_dark"] == 1.0)
                 & (joined["pass_id"] == p)).sum())
            for p in det["pass_id"].unique()
        ]
        estimate = estimate_dark_hours(counts, args.window_hours,
                                       dark_recall=args.dark_recall,
                                       bootstrap_iters=cfg.effort.bootstrap_iters)
        print(f"\n[{top}]")
        print(estimate.summary())
        with open(maps_dir / "methods_note.txt", "w") as f:
            f.write(f"MPA: {top}\n\n{estimate.summary()}\n")

    out_html = args.out or str(maps_dir / "pressure_map.html")
    out = build_pressure_map(joined, mpas, estimate, out_html)
    print(f"\nMap written to {out} — open it in a browser.")


if __name__ == "__main__":
    main()
