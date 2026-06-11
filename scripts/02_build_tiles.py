"""Step 2 — cut scenes into training chips (the tiling pipeline).

For every scene in the chosen split:
  * iterate overlapping tile windows (windowed rasterio reads — whole
    scenes are never loaded),
  * skip near-pure-land tiles,
  * keep every tile containing a labelled vessel; keep a small random
    fraction of empty-ocean tiles as negatives,
  * write each kept chip as .npz (image stack + point annotations),
  * write an index CSV (chips_<split>.csv) used by the training dataset.

Usage:
    python scripts/02_build_tiles.py --split train
    python scripts/02_build_tiles.py --split validation
    # add --max-scenes 5 for a quick small-scale first run
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from tqdm import tqdm

from dark_vessel.config import load_config
from dark_vessel.data.labels import labels_in_window, load_labels
from dark_vessel.data.tiling import SceneReader, iter_tile_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="limit scenes for a quick first run")
    ap.add_argument("--append", action="store_true",
                    help="keep existing index entries for scenes not present "
                         "on disk (batch workflow: download -> tile -> delete "
                         "raw scenes -> repeat)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg.tiling

    labels_key = "train_labels" if args.split == "train" else "val_labels"
    labels = load_labels(cfg.paths[labels_key], cfg.labels.train_confidences)

    split_dir = Path(cfg.paths.data_root) / args.split
    scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if args.max_scenes:
        scene_dirs = scene_dirs[: args.max_scenes]
    if not scene_dirs:
        raise SystemExit(f"No scene folders found in {split_dir}")

    out_dir = Path(cfg.paths.tiles_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.train.seed)
    index_rows = []

    for scene_dir in scene_dirs:
        reader = SceneReader(scene_dir, cfg)
        scene_labels = labels[labels["scene_id"] == reader.scene_id]
        specs = list(iter_tile_specs(reader.scene_id, reader.height,
                                     reader.width, t.tile_size, t.overlap))

        for spec in tqdm(specs, desc=reader.scene_id, leave=False):
            pts = labels_in_window(scene_labels, spec.row_off, spec.col_off,
                                   spec.height, spec.width)
            has_vessel = bool((pts["is_vessel"] == 1.0).any()) if len(pts) else False

            # Decide whether to keep this tile before paying the read cost.
            if not has_vessel and len(pts) == 0:
                if rng.random() > t.empty_tile_keep_prob:
                    continue

            mask = reader.read_mask(spec)
            if (mask > 0).mean() > t.max_land_fraction:
                continue

            image = reader.read_tile(spec)
            points = (
                pts[["tile_row", "tile_col", "is_vessel", "is_fishing",
                     "vessel_length_m"]].to_numpy(dtype=np.float32)
                if len(pts) else np.zeros((0, 5), dtype=np.float32)
            )

            chip_name = f"{spec.scene_id}_{spec.row_off}_{spec.col_off}.npz"
            chip_path = out_dir / chip_name
            np.savez_compressed(chip_path, image=image.astype(np.float16),
                                points=points)
            index_rows.append({
                "chip_path": str(chip_path),
                "scene_id": spec.scene_id,
                "row_off": spec.row_off,
                "col_off": spec.col_off,
                "n_points": len(points),
                "n_vessels": int((pts["is_vessel"] == 1.0).sum()) if len(pts) else 0,
            })
        reader.close()

    index = pd.DataFrame(index_rows)
    index_path = Path(cfg.paths.tiles_dir) / f"chips_{args.split}.csv"
    if args.append and index_path.exists():
        old = pd.read_csv(index_path)
        tiled_now = {d.name for d in scene_dirs}
        kept = old[~old["scene_id"].isin(tiled_now)]
        index = pd.concat([kept, index], ignore_index=True)
        print(f"--append: kept {len(kept)} chips from "
              f"{kept['scene_id'].nunique()} previously tiled scene(s)")
    index.to_csv(index_path, index=False)
    print(f"\nWrote {len(index):,} chips "
          f"({int((index['n_vessels'] > 0).sum()):,} contain vessels)")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
