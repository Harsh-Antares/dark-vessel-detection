"""PyTorch dataset for pre-cut training chips.

scripts/02_build_tiles.py writes each chip as a compressed .npz containing
the normalised image stack plus per-vessel point annotations, and an
index CSV listing every chip with its vessel count. This dataset:

  * renders the Gaussian center-heatmap target on the fly,
  * builds the per-point attribute targets (vessel / fishing / length)
    with NaN-aware masks so unknown attributes don't contribute to the loss,
  * applies simple flip/rotate augmentation (safe for SAR: the sea has no
    preferred orientation at this scale).

Positive-aware sampling: vessels are sparse, so a uniformly sampled batch
would be almost entirely empty ocean and the detector would learn to
predict "nothing" everywhere. PositiveAwareSampler draws a configurable
fraction of every epoch from chips that contain at least one vessel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

MAX_POINTS = 128  # per-chip cap; xView3 chips rarely exceed a few dozen


def gaussian_splat(heatmap: np.ndarray, row: float, col: float, sigma: float):
    """Draw a small 2-D Gaussian (peak = 1) centred at (row, col) in place.

    Overlapping vessels take the element-wise max, so nearby peaks stay
    distinct instead of summing into one blob.
    """
    radius = int(3 * sigma)
    h, w = heatmap.shape
    r0, c0 = int(round(row)), int(round(col))
    top, bottom = max(0, r0 - radius), min(h, r0 + radius + 1)
    left, right = max(0, c0 - radius), min(w, c0 + radius + 1)
    if top >= bottom or left >= right:
        return
    ys, xs = np.mgrid[top:bottom, left:right]
    g = np.exp(-((ys - row) ** 2 + (xs - col) ** 2) / (2 * sigma**2))
    np.maximum(heatmap[top:bottom, left:right], g,
               out=heatmap[top:bottom, left:right])


class ChipDataset(Dataset):
    def __init__(self, index_csv: str | Path, cfg, augment: bool = True):
        self.index = pd.read_csv(index_csv)
        self.cfg = cfg
        self.augment = augment
        self.sigma = cfg.model.heatmap_sigma
        # Relative chip paths are resolved against the index CSV's folder,
        # so a tile bundle works unchanged on any machine (laptop, Colab...).
        self.base = Path(index_csv).resolve().parent

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec = self.index.iloc[idx]
        chip_path = Path(rec["chip_path"])
        if not chip_path.is_absolute():
            chip_path = self.base / chip_path
        with np.load(chip_path) as z:
            image = z["image"].astype(np.float32)         # (C, H, W)
            points = z["points"].astype(np.float32)       # (N, 5): row, col, is_vessel, is_fishing, length_m

        if self.augment:
            image, points = self._augment(image, points)

        _, h, w = image.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        for p in points:
            gaussian_splat(heatmap, p[0], p[1], self.sigma)

        n = min(len(points), MAX_POINTS)
        pt_rows = np.zeros(MAX_POINTS, dtype=np.int64)
        pt_cols = np.zeros(MAX_POINTS, dtype=np.int64)
        vessel = np.zeros(MAX_POINTS, dtype=np.float32)
        fishing = np.zeros(MAX_POINTS, dtype=np.float32)
        length = np.zeros(MAX_POINTS, dtype=np.float32)
        vessel_mask = np.zeros(MAX_POINTS, dtype=np.float32)
        fishing_mask = np.zeros(MAX_POINTS, dtype=np.float32)
        length_mask = np.zeros(MAX_POINTS, dtype=np.float32)
        valid = np.zeros(MAX_POINTS, dtype=np.float32)

        for i, p in enumerate(points[:n]):
            pt_rows[i] = int(np.clip(round(p[0]), 0, h - 1))
            pt_cols[i] = int(np.clip(round(p[1]), 0, w - 1))
            valid[i] = 1.0
            if not np.isnan(p[2]):
                vessel[i], vessel_mask[i] = p[2], 1.0
            if not np.isnan(p[3]):
                fishing[i], fishing_mask[i] = p[3], 1.0
            if not np.isnan(p[4]) and p[4] > 0:
                # log-length keeps the regression well-scaled (lengths span
                # ~5 m skiffs to ~300 m tankers)
                length[i], length_mask[i] = np.log(p[4]), 1.0

        return {
            "image": torch.from_numpy(image),
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "pt_rows": torch.from_numpy(pt_rows),
            "pt_cols": torch.from_numpy(pt_cols),
            "valid": torch.from_numpy(valid),
            "vessel": torch.from_numpy(vessel),
            "vessel_mask": torch.from_numpy(vessel_mask),
            "fishing": torch.from_numpy(fishing),
            "fishing_mask": torch.from_numpy(fishing_mask),
            "length": torch.from_numpy(length),
            "length_mask": torch.from_numpy(length_mask),
        }

    def _augment(self, image: np.ndarray, points: np.ndarray):
        _, h, w = image.shape
        if np.random.rand() < 0.5:                         # horizontal flip
            image = image[:, :, ::-1].copy()
            if len(points):
                points = points.copy()
                points[:, 1] = w - 1 - points[:, 1]
        if np.random.rand() < 0.5:                         # vertical flip
            image = image[:, ::-1, :].copy()
            if len(points):
                points = points.copy()
                points[:, 0] = h - 1 - points[:, 0]
        k = np.random.randint(4)                           # 90-degree rotations
        if k and h == w:
            image = np.rot90(image, k, axes=(1, 2)).copy()
            if len(points):
                points = points.copy()
                for _ in range(k):
                    r = points[:, 0].copy()
                    points[:, 0] = w - 1 - points[:, 1]
                    points[:, 1] = r
        return image, points


class PositiveAwareSampler(Sampler):
    """Yields chip indices so ~positive_fraction of draws contain vessels."""

    def __init__(self, index: pd.DataFrame, positive_fraction: float,
                 num_samples: int | None = None, seed: int = 0):
        self.pos = np.flatnonzero(index["n_vessels"].values > 0)
        self.neg = np.flatnonzero(index["n_vessels"].values == 0)
        if len(self.pos) == 0:
            raise ValueError("No chips with vessels — check the tiling step.")
        self.frac = positive_fraction
        self.num_samples = num_samples or len(index)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_pos = int(round(self.num_samples * self.frac))
        n_neg = self.num_samples - n_pos
        pos = self.rng.choice(self.pos, size=n_pos, replace=True)
        if len(self.neg):
            neg = self.rng.choice(self.neg, size=n_neg, replace=True)
            out = np.concatenate([pos, neg])
        else:
            out = pos
        self.rng.shuffle(out)
        return iter(out.tolist())
