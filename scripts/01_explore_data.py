"""Step 1 — sanity-check the downloaded xView3 data and labels.

Prints per-split label statistics (vessel / fishing / dark counts, length
distribution, confidence grades) and, for one scene, the raster layout.
Run this first: if it's happy, the rest of the pipeline will be too.

Usage:
    python scripts/01_explore_data.py [--config configs/default.yaml] [--scene SCENE_ID]
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from dark_vessel.config import load_config
from dark_vessel.data.labels import load_labels


def describe_labels(csv_path: str, name: str):
    print(f"\n===== {name} labels: {csv_path} =====")
    raw = pd.read_csv(csv_path)
    print(f"rows: {len(raw):,}   scenes: {raw['scene_id'].nunique()}")
    print("\nconfidence grades:")
    print(raw["confidence"].value_counts().to_string())

    df = load_labels(csv_path)
    vessels = df[df["is_vessel"] == 1.0]
    print(f"\nafter HIGH/MEDIUM filter: {len(df):,} rows, {len(vessels):,} vessels")
    print(f"fishing vessels: {int((df['is_fishing'] == 1.0).sum()):,}")
    print(f"DARK vessels (no AIS correlation): {int((df['is_dark'] == 1.0).sum()):,}")
    print(f"COOPERATIVE vessels (AIS matched): {int((df['is_dark'] == 0.0).sum()):,}")
    lengths = vessels["vessel_length_m"].dropna()
    if len(lengths):
        print(f"length (m): median={lengths.median():.0f}  "
              f"p5={lengths.quantile(0.05):.0f}  p95={lengths.quantile(0.95):.0f}")
    return df


def describe_scene(scene_dir: Path, cfg):
    import rasterio

    print(f"\n===== scene: {scene_dir.name} =====")
    for name in list(cfg.scene.channels) + list(cfg.scene.ancillary) + [cfg.scene.mask_file]:
        path = scene_dir / name
        if not path.exists():
            print(f"  MISSING: {name}")
            continue
        with rasterio.open(path) as src:
            print(f"  {name}: {src.height} x {src.width}  crs={src.crs}  "
                  f"dtype={src.dtypes[0]}  nodata={src.nodata}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scene", default=None, help="scene_id to inspect")
    args = ap.parse_args()
    cfg = load_config(args.config)

    for split, key in (("train", "train_labels"), ("validation", "val_labels")):
        path = cfg.paths[key]
        if Path(path).exists():
            describe_labels(path, split)
        else:
            print(f"\n[skip] {split} labels not found at {path}")

    data_root = Path(cfg.paths.data_root)
    if args.scene:
        for split in ("train", "validation"):
            d = data_root / split / args.scene
            if d.exists():
                describe_scene(d, cfg)
                break
        else:
            print(f"scene {args.scene} not found under {data_root}")
    else:
        # Inspect the first scene we can find.
        for split in ("validation", "train"):
            split_dir = data_root / split
            if split_dir.exists():
                scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
                if scenes:
                    describe_scene(scenes[0], cfg)
                    break


if __name__ == "__main__":
    main()
