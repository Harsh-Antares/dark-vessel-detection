"""Pack training tiles into one portable archive for Google Colab.

Creates outputs/colab_tiles.tar containing:
    tiles/train/<every training chip>.npz
    tiles/validation/<a stratified sample of validation chips>.npz
    tiles/chips_train.csv         (chip paths RELATIVE to the csv)
    tiles/chips_validation.csv

Why a sample of validation: the full validation split is ~18 GB — far too
big to upload to Drive, and per-epoch validation only uses a few hundred
chips anyway (cfg.train.val_max_chips). The sample is stratified
half-with-vessels / half-empty with a fixed seed, so numbers stay
comparable between runs. The FULL evaluation still happens locally on
whole scenes (scripts 04/05).

Usage:
    python scripts/make_colab_bundle.py                # default 600 val chips
    python scripts/make_colab_bundle.py --val-chips 800
Then upload outputs/colab_tiles.tar to Google Drive (folder: darkvessel/)
and open notebooks/train_colab.ipynb in Colab.
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
from tqdm import tqdm

from dark_vessel.config import load_config


def add_split(tar, index_csv: Path, split: str, sample: int | None, seed: int):
    df = pd.read_csv(index_csv)
    if sample and len(df) > sample:
        pos = df[df["n_vessels"] > 0]
        neg = df[df["n_vessels"] == 0]
        n_pos = min(len(pos), sample // 2)
        n_neg = min(len(neg), sample - n_pos)
        df = pd.concat([pos.sample(n_pos, random_state=seed),
                        neg.sample(n_neg, random_state=seed)]
                       ).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"{split}: sampled {len(df)} chips "
              f"({n_pos} with vessels, {n_neg} empty)")
    else:
        print(f"{split}: taking all {len(df)} chips")

    rel_paths = []
    total_bytes = 0
    for chip in tqdm(df["chip_path"], desc=f"packing {split}"):
        src = Path(chip)
        arc = f"tiles/{split}/{src.name}"
        tar.add(src, arcname=arc)
        rel_paths.append(f"{split}/{src.name}")
        total_bytes += src.stat().st_size

    df = df.copy()
    df["chip_path"] = rel_paths
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        tar.add(f.name, arcname=f"tiles/chips_{split}.csv")
    return total_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--val-chips", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    tiles = Path(cfg.paths.tiles_dir)
    train_csv = tiles / "chips_train.csv"
    val_csv = tiles / "chips_validation.csv"
    for p in (train_csv, val_csv):
        if not p.exists():
            raise SystemExit(f"Missing {p} — run scripts/02_build_tiles.py first.")

    out = Path(args.out) if args.out else tiles.parent / "colab_tiles.tar"
    total = 0
    with tarfile.open(out, "w") as tar:  # uncompressed: chips are already npz-compressed
        total += add_split(tar, train_csv, "train", None, cfg.train.seed)
        total += add_split(tar, val_csv, "validation", args.val_chips, cfg.train.seed)

    print(f"\nBundle written: {out}  ({out.stat().st_size / 1e9:.2f} GB)")
    print("Next: upload it to Google Drive (create a folder called 'darkvessel'),")
    print("then open notebooks/train_colab.ipynb in Google Colab.")


if __name__ == "__main__":
    main()
