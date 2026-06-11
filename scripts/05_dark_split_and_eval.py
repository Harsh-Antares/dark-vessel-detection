"""Step 5 — the dark/cooperative split + the full evaluation suite.

For each scene's detection CSV:
  * transfer dark/cooperative status from the xView3 ground truth
    (AIS-correlation result encoded in the confidence field),
  * compute the metric breakdown: detection F1, DARK RECALL, near-shore
    recall, vessel/fishing classification F1, length score, aggregate.

Writes <scene>_detections_dark.csv (detections + is_dark column) and
prints the evaluation report.

Usage:
    python scripts/05_dark_split_and_eval.py --split validation
    python scripts/05_dark_split_and_eval.py --split validation --scene <SCENE_ID>
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from dark_vessel.config import load_config
from dark_vessel.data.labels import load_labels
from dark_vessel.evaluation.metrics import evaluate_scene, format_report
from dark_vessel.fusion.ais_correlation import transfer_dark_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["train", "validation"], default="validation")
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    labels_key = "train_labels" if args.split == "train" else "val_labels"
    truth_all = load_labels(cfg.paths[labels_key], cfg.labels.train_confidences)

    pred_dir = Path(cfg.paths.predictions_dir)
    csvs = sorted(pred_dir.glob("*_detections.csv"))
    if args.scene:
        csvs = [c for c in csvs if c.name.startswith(args.scene)]
    if not csvs:
        raise SystemExit(f"No detection CSVs in {pred_dir} — run 04_predict_scene.py first.")

    reports = {}
    for csv in csvs:
        scene_id = csv.name.replace("_detections.csv", "")
        det = pd.read_csv(csv)
        truth = truth_all[truth_all["scene_id"] == scene_id]
        if truth.empty:
            print(f"[skip] no {args.split} labels for scene {scene_id}")
            continue

        det = transfer_dark_labels(det, truth, cfg.inference.match_radius_m)
        out_csv = pred_dir / f"{scene_id}_detections_dark.csv"
        det.to_csv(out_csv, index=False)

        report = evaluate_scene(det, truth,
                                pixel_spacing_m=cfg.scene.pixel_spacing_m,
                                match_radius_m=cfg.inference.match_radius_m)
        reports[scene_id] = report
        n_dark = int((det["is_dark"] == 1.0).sum())
        n_coop = int((det["is_dark"] == 0.0).sum())
        print(f"\n### {scene_id}  (dark={n_dark}, cooperative={n_coop}) -> {out_csv.name}")
        print(format_report(report))

    out_json = pred_dir / f"evaluation_{args.split}.json"
    with open(out_json, "w") as f:
        json.dump(reports, f, indent=2, default=float)
    print(f"\nFull reports saved to {out_json}")


if __name__ == "__main__":
    main()
