"""Geographic non-maximum suppression.

Tiles overlap by design (so no vessel is cut in half by a seam), which
means a vessel sitting in the overlap strip is detected twice — once per
tile. Pixel-space NMS inside one tile can't fix that; the duplicates live
in *different* tiles. So we suppress in scene/geographic coordinates:
greedy NMS on the full detection list, keeping the highest-scoring
detection and removing everything within `radius` of it.

A cKDTree makes the neighbour lookup O(N log N) instead of O(N^2).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def geographic_nms(rows: np.ndarray, cols: np.ndarray, scores: np.ndarray,
                   radius_px: float) -> np.ndarray:
    """Greedy radius-NMS in scene pixel coordinates.

    Returns the indices of detections to KEEP, ordered by descending score.
    """
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    pts = np.stack([rows, cols], axis=1)
    tree = cKDTree(pts)
    order = np.argsort(-scores)
    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        for j in tree.query_ball_point(pts[idx], r=radius_px):
            if j != idx:
                suppressed[j] = True
    return np.asarray(keep, dtype=np.int64)
