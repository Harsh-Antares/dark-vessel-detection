"""Heatmap peak extraction.

A detection is a local maximum of the predicted heatmap above a score
threshold. The classic fast implementation: a 3x3 max-filter; a pixel is a
peak iff it equals the filtered value (it is the max of its neighbourhood).
This replaces all anchor/NMS machinery of box detectors with five lines.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def extract_peaks(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Return (N, 3) array of [row, col, score] local maxima above threshold."""
    local_max = maximum_filter(heatmap, size=3, mode="constant") == heatmap
    mask = local_max & (heatmap >= threshold)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    scores = heatmap[rows, cols]
    order = np.argsort(-scores)
    return np.stack([rows[order], cols[order], scores[order]],
                    axis=1).astype(np.float32)
