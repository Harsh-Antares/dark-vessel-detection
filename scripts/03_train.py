"""Step 3 — train the center-heatmap detector with attribute heads.

Prerequisite: scripts/02_build_tiles.py has produced
    outputs/tiles/chips_train.csv  and  outputs/tiles/chips_validation.csv

Usage:
    python scripts/03_train.py
    python scripts/03_train.py --epochs 5        # quick smoke run
Checkpoints land in outputs/checkpoints/ (best.pt = highest val F1).
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from dark_vessel.config import load_config
from dark_vessel.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size

    tiles = Path(cfg.paths.tiles_dir)
    train_index = tiles / "chips_train.csv"
    val_index = tiles / "chips_validation.csv"
    for p in (train_index, val_index):
        if not p.exists():
            raise SystemExit(f"Missing {p} — run scripts/02_build_tiles.py first.")

    train(cfg, str(train_index), str(val_index))


if __name__ == "__main__":
    main()
