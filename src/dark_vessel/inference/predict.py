"""Full-scene inference: sliding window -> peaks -> geographic NMS -> CSV.

For each tile we also read the model's attribute maps at every peak, giving
each detection: vessel probability, fishing probability, estimated length.
Detections on land (per the scene mask) are dropped — a hard post-filter
that kills most coastal false positives for free.

Output: a pandas DataFrame / CSV with one row per detection:
    scene_id, scene_row, scene_col, lon, lat, score,
    p_vessel, p_fishing, length_m
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.tiling import SceneReader, TileSpec, iter_tile_specs
from .nms import geographic_nms
from .peaks import extract_peaks


def load_model(checkpoint_path: str | Path, cfg, device):
    from ..models.model import DarkVesselNet

    model = DarkVesselNet(cfg).to(device)
    # weights_only=False: the checkpoint embeds our config dict and metric
    # floats alongside the weights, and we produced the file ourselves.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_scene(model, scene_dir: str | Path, cfg, device) -> pd.DataFrame:
    reader = SceneReader(scene_dir, cfg)
    t = cfg.tiling
    specs = list(iter_tile_specs(reader.scene_id, reader.height, reader.width,
                                 t.tile_size, t.overlap))

    all_rows, all_cols, all_scores = [], [], []
    all_vessel, all_fishing, all_length = [], [], []

    for spec in tqdm(specs, desc=f"scene {reader.scene_id}"):
        mask = reader.read_mask(spec)
        if (mask > 0).mean() > t.max_land_fraction:
            continue  # almost pure land — nothing to detect

        image = reader.read_tile(spec)
        x = torch.from_numpy(image).unsqueeze(0).to(device)
        out = model(x)
        heat = torch.sigmoid(out["heatmap"])[0, 0].cpu().numpy()

        # Hard land filter: zero the heatmap wherever the mask says land/ice.
        heat[mask > 0] = 0.0

        peaks = extract_peaks(heat, cfg.inference.score_threshold)
        if len(peaks) == 0:
            continue

        p_vessel = torch.sigmoid(out["vessel"])[0, 0].cpu().numpy()
        p_fishing = torch.sigmoid(out["fishing"])[0, 0].cpu().numpy()
        log_len = out["length"][0, 0].cpu().numpy()

        r = peaks[:, 0].astype(int)
        c = peaks[:, 1].astype(int)
        all_rows.append(peaks[:, 0] + spec.row_off)
        all_cols.append(peaks[:, 1] + spec.col_off)
        all_scores.append(peaks[:, 2])
        all_vessel.append(p_vessel[r, c])
        all_fishing.append(p_fishing[r, c])
        all_length.append(np.exp(log_len[r, c]))

    reader.close()
    if not all_rows:
        return pd.DataFrame(columns=["scene_id", "scene_row", "scene_col",
                                     "lon", "lat", "score", "p_vessel",
                                     "p_fishing", "length_m"])

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    scores = np.concatenate(all_scores)
    vessel = np.concatenate(all_vessel)
    fishing = np.concatenate(all_fishing)
    length = np.concatenate(all_length)

    # De-duplicate across tile seams in scene coordinates.
    radius_px = cfg.inference.nms_radius_m / cfg.scene.pixel_spacing_m
    keep = geographic_nms(rows, cols, scores, radius_px)

    lon, lat = reader.pixel_to_lonlat(rows[keep], cols[keep])
    return pd.DataFrame({
        "scene_id": reader.scene_id,
        "scene_row": rows[keep],
        "scene_col": cols[keep],
        "lon": lon,
        "lat": lat,
        "score": scores[keep],
        "p_vessel": vessel[keep],
        "p_fishing": fishing[keep],
        "length_m": length[keep],
    })
