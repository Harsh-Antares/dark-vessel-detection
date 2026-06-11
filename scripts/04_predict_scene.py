"""Step 4 — run full-scene inference.

Slides the trained model over a whole scene (windowed reads), extracts
heatmap peaks, applies the land filter and geographic NMS, and writes one
CSV of detections per scene to outputs/predictions/.

Usage:
    python scripts/04_predict_scene.py --scene <SCENE_ID> --split validation
    python scripts/04_predict_scene.py --all --split validation
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from dark_vessel.config import load_config
from dark_vessel.inference.predict import load_model, predict_scene
from dark_vessel.train import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to outputs/checkpoints/best.pt")
    ap.add_argument("--split", choices=["train", "validation"], default="validation")
    ap.add_argument("--scene", default=None, help="single scene_id")
    ap.add_argument("--all", action="store_true", help="run every scene in the split")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ckpt = args.checkpoint or str(Path(cfg.paths.checkpoints_dir) / "best.pt")
    if not Path(ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {ckpt} — train first (03_train.py).")

    device = pick_device()
    model = load_model(ckpt, cfg, device)

    split_dir = Path(cfg.paths.data_root) / args.split
    if args.scene:
        scene_dirs = [split_dir / args.scene]
    elif args.all:
        scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    else:
        raise SystemExit("Pass --scene SCENE_ID or --all")

    out_dir = Path(cfg.paths.predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"[skip] {scene_dir} does not exist")
            continue
        det = predict_scene(model, scene_dir, cfg, device)
        out_csv = out_dir / f"{scene_dir.name}_detections.csv"
        det.to_csv(out_csv, index=False)
        print(f"{scene_dir.name}: {len(det)} detections -> {out_csv}")


if __name__ == "__main__":
    main()
