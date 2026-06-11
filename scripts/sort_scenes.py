"""Helper — sort downloaded/extracted xView3 scene folders into splits.

The xView3 downloader drops every scene archive into one folder, and the
tiny set even mixes training and validation scenes. This script looks up
each extracted scene folder's ID in the label CSVs and MOVES it to:

    data/xview3/train/<scene_id>        if it appears in train.csv
    data/xview3/validation/<scene_id>   if it appears in validation.csv

Scenes in neither CSV (e.g. public-leaderboard scenes) are left where
they are and reported.

Usage:
    python scripts/sort_scenes.py                      # default source dir
    python scripts/sort_scenes.py --source data/xview3/downloads
"""

import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from dark_vessel.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default=None,
                    help="folder containing extracted scene folders "
                         "(default: <data_root>/downloads)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_root = Path(cfg.paths.data_root)
    source = Path(args.source) if args.source else data_root / "downloads"
    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")

    split_ids = {}
    for split, key in (("train", "train_labels"), ("validation", "val_labels")):
        csv = Path(cfg.paths[key])
        if csv.exists():
            split_ids[split] = set(pd.read_csv(csv)["scene_id"].unique())
        else:
            print(f"[warn] {csv} not found — cannot sort {split} scenes")
            split_ids[split] = set()

    moved, unknown = 0, []
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        dest_split = next((s for s, ids in split_ids.items()
                           if folder.name in ids), None)
        if dest_split is None:
            unknown.append(folder.name)
            continue
        dest = data_root / dest_split / folder.name
        if dest.exists():
            print(f"[skip] {folder.name} already in {dest_split}/")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(dest))
        print(f"{folder.name}  ->  {dest_split}/")
        moved += 1

    print(f"\nMoved {moved} scene folder(s).")
    if unknown:
        print(f"Left in place (not in either label CSV): {unknown}")


if __name__ == "__main__":
    main()
