"""xView3 label handling and the dark/cooperative semantics.

The label CSVs contain one row per detection with (among others):

    scene_id, detect_lat, detect_lon,
    detect_scene_row, detect_scene_column,   # pixel position in the scene
    is_vessel, is_fishing, vessel_length_m,
    distance_from_shore_km, confidence       # HIGH / MEDIUM / LOW

The crucial fact (stated in the project brief): xView3 labels were produced
by matching AIS broadcasts to SAR detections.

    * confidence == HIGH   -> the detection correlated with an AIS track
                              (identity known)            => COOPERATIVE
    * confidence == MEDIUM -> radar-visible object with NO AIS correlation
                              (visually confirmed)        => DARK candidate
    * confidence == LOW    -> uncertain; excluded from training

"Dark" is therefore *not* a visual class — a dark trawler and a cooperative
trawler look identical in radar. It is the result of data fusion, and we
inherit that fusion result through the confidence field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "scene_id",
    "detect_scene_row",
    "detect_scene_column",
    "is_vessel",
    "is_fishing",
    "vessel_length_m",
    "confidence",
]


def load_labels(csv_path: str | Path, confidences=("HIGH", "MEDIUM")) -> pd.DataFrame:
    """Load an xView3 label CSV and attach the derived `is_dark` column."""
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}")

    df = df[df["confidence"].isin(list(confidences))].copy()

    # Booleans arrive as a mix of True/False, strings and NaN; normalise.
    for col in ("is_vessel", "is_fishing"):
        df[col] = df[col].map(
            {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
        ).astype(float)  # NaN stays NaN = "unknown", masked out of the loss

    # The dark/cooperative ground truth (see module docstring).
    df["is_dark"] = np.where(
        (df["confidence"] == "MEDIUM") & (df["is_vessel"] == 1.0), 1.0, 0.0
    )
    # Only vessels can meaningfully be dark; non-vessels get NaN.
    df.loc[df["is_vessel"] != 1.0, "is_dark"] = np.nan

    df["vessel_length_m"] = pd.to_numeric(df["vessel_length_m"], errors="coerce")
    return df


def labels_in_window(df: pd.DataFrame, row_off: int, col_off: int,
                     height: int, width: int) -> pd.DataFrame:
    """Subset of labels falling inside a tile window, with tile-local coords."""
    inside = df[
        (df["detect_scene_row"] >= row_off)
        & (df["detect_scene_row"] < row_off + height)
        & (df["detect_scene_column"] >= col_off)
        & (df["detect_scene_column"] < col_off + width)
    ].copy()
    inside["tile_row"] = inside["detect_scene_row"] - row_off
    inside["tile_col"] = inside["detect_scene_column"] - col_off
    return inside
