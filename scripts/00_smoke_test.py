"""Step 0 — end-to-end smoke test on SYNTHETIC data (no download needed).

Generates a small fake "scene" (SAR-like noise with bright vessel blobs +
GeoTIFF georeferencing), fake labels in xView3 CSV format, then runs the
real pipeline: tiling -> dataset -> 2 training epochs -> full-scene
inference -> dark split -> evaluation. If this passes, the code is wired
correctly and only the real data is missing.

Usage:
    python scripts/00_smoke_test.py
"""

import shutil
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from dark_vessel.config import DotDict, load_config

SCENE_ID = "smoke_scene_1"
SCENE_SIZE = 768          # small synthetic scene
N_VESSELS = 25


def make_synthetic_scene(root: Path, rng):
    scene_dir = root / "validation" / SCENE_ID
    scene_dir.mkdir(parents=True)
    transform = from_origin(12.0, 45.0, 0.0001, 0.0001)  # fake lon/lat grid

    # Sea clutter: gamma-distributed backscatter, converted to dB.
    base = 10 * np.log10(rng.gamma(1.5, 0.02, (SCENE_SIZE, SCENE_SIZE)))

    rows = rng.integers(40, SCENE_SIZE - 40, N_VESSELS)
    cols = rng.integers(40, SCENE_SIZE - 40, N_VESSELS)
    vh = base.copy()
    for r, c in zip(rows, cols):
        vh[r - 2:r + 3, c - 2:c + 3] += 25.0  # bright 5x5 vessel blob
    vv = vh + rng.normal(2.0, 0.5, vh.shape)

    layers = {
        "VH_dB.tif": vh.astype(np.float32),
        "VV_dB.tif": vv.astype(np.float32),
        "bathymetry.tif": np.full_like(vh, -200.0, dtype=np.float32),
        "owiWindSpeed.tif": np.full_like(vh, 6.0, dtype=np.float32),
        "owiMask.tif": np.zeros_like(vh, dtype=np.float32),
    }
    for name, arr in layers.items():
        with rasterio.open(
            scene_dir / name, "w", driver="GTiff",
            height=SCENE_SIZE, width=SCENE_SIZE, count=1,
            dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(arr, 1)

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    labels = pd.DataFrame({
        "scene_id": SCENE_ID,
        "detect_scene_row": rows,
        "detect_scene_column": cols,
        "detect_lon": xs,
        "detect_lat": ys,
        "is_vessel": True,
        "is_fishing": rng.random(N_VESSELS) > 0.5,
        "vessel_length_m": rng.uniform(15, 80, N_VESSELS).round(1),
        "distance_from_shore_km": rng.uniform(5, 50, N_VESSELS).round(1),
        # ~40% dark (MEDIUM = no AIS correlation)
        "confidence": np.where(rng.random(N_VESSELS) < 0.4, "MEDIUM", "HIGH"),
    })
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    labels.to_csv(labels_dir / "validation.csv", index=False)
    return scene_dir


def main():
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp(prefix="darkvessel_smoke_"))
    print(f"workspace: {tmp}")
    try:
        cfg = load_config("configs/default.yaml")
        # Shrink everything for the smoke run.
        cfg.paths.data_root = str(tmp / "data")
        cfg.paths.val_labels = str(tmp / "data/labels/validation.csv")
        cfg.paths.train_labels = cfg.paths.val_labels
        cfg.paths.tiles_dir = str(tmp / "tiles")
        cfg.paths.checkpoints_dir = str(tmp / "ckpt")
        cfg.tiling = DotDict({**cfg.tiling, "tile_size": 256, "overlap": 32,
                              "empty_tile_keep_prob": 0.3})
        cfg.model = DotDict({**cfg.model, "encoder": "resnet18",
                             "pretrained": False})
        # Enough steps for the BatchNorm running statistics to settle —
        # with very few optimiser steps the model looks fine in train mode
        # but collapses in eval mode.
        cfg.train = DotDict({**cfg.train, "epochs": 30, "batch_size": 4,
                             "num_workers": 0, "mixed_precision": False})
        cfg.inference = DotDict({**cfg.inference, "score_threshold": 0.2})

        print("\n[1/5] synthesising scene...")
        scene_dir = make_synthetic_scene(Path(cfg.paths.data_root), rng)

        print("[2/5] tiling...")
        from dark_vessel.data.labels import labels_in_window, load_labels
        from dark_vessel.data.tiling import SceneReader, iter_tile_specs

        labels = load_labels(cfg.paths.val_labels)
        reader = SceneReader(scene_dir, cfg)
        out_dir = Path(cfg.paths.tiles_dir) / "validation"
        out_dir.mkdir(parents=True)
        index_rows = []
        for spec in iter_tile_specs(SCENE_ID, reader.height, reader.width,
                                    cfg.tiling.tile_size, cfg.tiling.overlap):
            pts = labels_in_window(labels, spec.row_off, spec.col_off,
                                   spec.height, spec.width)
            image = reader.read_tile(spec)
            points = (pts[["tile_row", "tile_col", "is_vessel", "is_fishing",
                           "vessel_length_m"]].to_numpy(np.float32)
                      if len(pts) else np.zeros((0, 5), np.float32))
            chip = out_dir / f"{SCENE_ID}_{spec.row_off}_{spec.col_off}.npz"
            np.savez_compressed(chip, image=image.astype(np.float16),
                                points=points)
            index_rows.append({"chip_path": str(chip), "n_points": len(points),
                               "n_vessels": int((pts["is_vessel"] == 1).sum())
                               if len(pts) else 0})
        reader.close()
        index = pd.DataFrame(index_rows)
        index_csv = Path(cfg.paths.tiles_dir) / "chips_validation.csv"
        index.to_csv(index_csv, index=False)
        print(f"   {len(index)} chips, "
              f"{int((index['n_vessels'] > 0).sum())} with vessels")

        print(f"[3/5] training {cfg.train.epochs} epochs...")
        from dark_vessel.train import train
        train(cfg, str(index_csv), str(index_csv))

        print("[4/5] full-scene inference...")
        import torch

        from dark_vessel.inference.predict import load_model, predict_scene
        from dark_vessel.train import pick_device
        device = pick_device()
        model = load_model(Path(cfg.paths.checkpoints_dir) / "best.pt", cfg, device)
        det = predict_scene(model, scene_dir, cfg, device)
        print(f"   {len(det)} detections (truth: {N_VESSELS})")

        print("[5/5] dark split + evaluation...")
        from dark_vessel.evaluation.metrics import evaluate_scene, format_report
        from dark_vessel.fusion.ais_correlation import transfer_dark_labels
        det = transfer_dark_labels(det, labels, cfg.inference.match_radius_m)
        report = evaluate_scene(det, labels)
        print(format_report(report))

        print("\nSMOKE TEST PASSED — the pipeline is wired correctly.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
