"""Stage 4 — SAR x AIS data fusion: the dark/cooperative split.

THE HONEST TECHNICAL NOTE (state this everywhere): you cannot tell from a
SAR chip whether a vessel is broadcasting AIS. A dark trawler and a
cooperative trawler scatter microwaves identically. "Dark" is therefore
NOT a visual class the network predicts — it is the *result of a fusion
step*: a SAR detection that fails to correlate with any AIS message within
a space-time matching window is dark.

Two operating modes:

1. xView3 mode (what the scripts use): the dataset's labels were built by
   exactly this AIS correlation, encoded in the `confidence` field
   (HIGH = AIS-correlated => cooperative; MEDIUM = no correlation => dark).
   We transfer that status to our detections by nearest-neighbour matching
   detections to ground-truth points.

2. Live mode: given an external AIS table (timestamped positions), we
   interpolate every track to the SAR acquisition time and match each
   detection within a radius. No match => dark. This is what a real
   deployment would run with an open AIS feed.

Optional ML flourish: `fit_dark_prior` trains a calibrated logistic
regression P(dark | length, fishing prob., distance from shore) — a *risk
score for triage*, explicitly framed as a prior, never as an identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_000.0


def _lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, lat0: float):
    """Local equirectangular projection (metres) — fine for <100 km extents."""
    x = np.radians(lon) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat) * EARTH_RADIUS_M
    return x, y


def transfer_dark_labels(detections: pd.DataFrame, truth: pd.DataFrame,
                         match_radius_m: float = 200.0) -> pd.DataFrame:
    """xView3 mode: inherit dark/cooperative status from matched GT points.

    detections: needs lon/lat columns (from predict_scene).
    truth: the label dataframe from labels.load_labels for the same scene,
           with detect_lat/detect_lon and is_dark.

    Adds columns: matched_gt (bool), is_dark (1/0/NaN).
    Unmatched detections get is_dark = NaN ("unknown" — they may be false
    positives or unlabelled targets).
    """
    det = detections.copy()
    det["matched_gt"] = False
    det["is_dark"] = np.nan
    if len(det) == 0 or len(truth) == 0:
        return det

    lat0 = float(det["lat"].mean())
    dx, dy = _lonlat_to_xy(det["lon"].values, det["lat"].values, lat0)
    tx, ty = _lonlat_to_xy(truth["detect_lon"].values,
                           truth["detect_lat"].values, lat0)

    tree = cKDTree(np.stack([tx, ty], axis=1))
    dist, idx = tree.query(np.stack([dx, dy], axis=1),
                           distance_upper_bound=match_radius_m)
    hit = np.isfinite(dist)
    det.loc[hit, "matched_gt"] = True
    det.loc[hit, "is_dark"] = truth["is_dark"].values[idx[hit]]
    return det


def correlate_with_ais(detections: pd.DataFrame, ais: pd.DataFrame,
                       scene_time: pd.Timestamp,
                       match_radius_m: float = 500.0,
                       max_gap: pd.Timedelta = pd.Timedelta("30min"),
                       ) -> pd.DataFrame:
    """Live mode: interpolate AIS tracks to scene_time and match detections.

    ais columns required: mmsi, timestamp (UTC), lon, lat.
    Adds: is_dark (1 = no AIS match, 0 = matched), matched_mmsi.
    """
    det = detections.copy()
    det["is_dark"] = 1.0
    det["matched_mmsi"] = pd.NA
    if len(det) == 0:
        return det

    ais = ais.copy()
    ais["timestamp"] = pd.to_datetime(ais["timestamp"], utc=True)
    scene_time = pd.Timestamp(scene_time).tz_convert("UTC") \
        if pd.Timestamp(scene_time).tzinfo else pd.Timestamp(scene_time, tz="UTC")

    # Interpolate each vessel's track to the SAR acquisition instant.
    interp_rows = []
    for mmsi, track in ais.groupby("mmsi"):
        track = track.sort_values("timestamp")
        before = track[track["timestamp"] <= scene_time].tail(1)
        after = track[track["timestamp"] >= scene_time].head(1)
        if len(before) and len(after):
            b, a = before.iloc[0], after.iloc[0]
            span = (a["timestamp"] - b["timestamp"]).total_seconds()
            if span == 0:
                lon, lat = b["lon"], b["lat"]
            else:
                w = (scene_time - b["timestamp"]).total_seconds() / span
                lon = b["lon"] + w * (a["lon"] - b["lon"])
                lat = b["lat"] + w * (a["lat"] - b["lat"])
            interp_rows.append((mmsi, lon, lat))
        elif len(before) and scene_time - before.iloc[0]["timestamp"] <= max_gap:
            interp_rows.append((mmsi, before.iloc[0]["lon"], before.iloc[0]["lat"]))
        elif len(after) and after.iloc[0]["timestamp"] - scene_time <= max_gap:
            interp_rows.append((mmsi, after.iloc[0]["lon"], after.iloc[0]["lat"]))

    if not interp_rows:
        return det  # no usable AIS -> everything stays dark (and say so!)

    ais_now = pd.DataFrame(interp_rows, columns=["mmsi", "lon", "lat"])
    lat0 = float(det["lat"].mean())
    dx, dy = _lonlat_to_xy(det["lon"].values, det["lat"].values, lat0)
    ax, ay = _lonlat_to_xy(ais_now["lon"].values, ais_now["lat"].values, lat0)

    tree = cKDTree(np.stack([ax, ay], axis=1))
    dist, idx = tree.query(np.stack([dx, dy], axis=1),
                           distance_upper_bound=match_radius_m)
    hit = np.isfinite(dist)
    det.loc[hit, "is_dark"] = 0.0
    det.loc[hit, "matched_mmsi"] = ais_now["mmsi"].values[idx[hit]]
    return det


def fit_dark_prior(detections: pd.DataFrame):
    """Train a calibrated P(dark | length, fishing, score) risk classifier.

    Returns (model, feature_names). Use model.predict_proba(X)[:, 1] as a
    triage ranking. This is a PRIOR for prioritising patrol effort —
    it does not and cannot identify a specific vessel as dark.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    labelled = detections.dropna(subset=["is_dark"])
    features = ["length_m", "p_fishing", "score"]
    X = labelled[features].fillna(labelled[features].median()).values
    y = labelled["is_dark"].astype(int).values
    if len(np.unique(y)) < 2:
        raise ValueError("Need both dark and cooperative examples to fit.")

    base = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X, y)
    return model, features
