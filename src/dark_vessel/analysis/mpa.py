"""Stage 5a — spatial join of detections to marine protected areas.

MPA polygons come from the World Database on Protected Areas (WDPA /
Protected Planet). Download the marine subset for your region as a
GeoPackage or shapefile and point cfg.paths.wdpa_path at it.

Outputs per MPA:
    n_total, n_dark, n_cooperative, dark_ratio,
    area_km2, dark_density_per_km2 (dark vessels per km^2 per SAR pass).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def load_mpas(wdpa_path, bbox=None) -> gpd.GeoDataFrame:
    """Load WDPA polygons, optionally clipped to (minx, miny, maxx, maxy).

    wdpa_path may be a single file or a list of files (e.g. the Italy and
    Croatia country downloads for an Adriatic scene) — they are simply
    concatenated.
    """
    paths = wdpa_path if isinstance(wdpa_path, (list, tuple)) else [wdpa_path]
    frames = [gpd.read_file(p, bbox=bbox) for p in paths]
    gdf = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry").to_crs("EPSG:4326")
    name_col = next((c for c in ("NAME", "name", "ORIG_NAME") if c in gdf.columns), None)
    if name_col is None:
        gdf["NAME"] = [f"MPA_{i}" for i in range(len(gdf))]
        name_col = "NAME"
    gdf = gdf.rename(columns={name_col: "mpa_name"})
    return gdf[["mpa_name", "geometry"]]


def detections_to_gdf(det: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        det.copy(),
        geometry=gpd.points_from_xy(det["lon"], det["lat"]),
        crs="EPSG:4326",
    )


def join_detections_to_mpas(det: pd.DataFrame,
                            mpas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Tag each detection with the MPA it falls inside (NaN if outside all)."""
    gdet = detections_to_gdf(det)
    joined = gpd.sjoin(gdet, mpas, how="left", predicate="within")
    return joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])


def mpa_summary(joined: gpd.GeoDataFrame, mpas: gpd.GeoDataFrame,
                n_passes: int = 1) -> pd.DataFrame:
    """Per-MPA dark-vessel pressure table.

    Areas are computed in an equal-area projection (World Mollweide) —
    computing km^2 in lon/lat degrees would be meaningless.
    """
    area_km2 = mpas.to_crs("ESRI:54009").geometry.area / 1e6
    # One MPA can span several WDPA polygons (zones); sum them per name.
    areas = (pd.Series(area_km2.values, index=mpas["mpa_name"].values)
             .groupby(level=0).sum())

    rows = []
    inside = joined.dropna(subset=["mpa_name"])
    # A detection inside two overlapping zones of the same MPA must count once.
    inside = (inside.reset_index()
              .drop_duplicates(subset=["index", "mpa_name"])
              .set_index("index"))
    for name, grp in inside.groupby("mpa_name"):
        is_dark = grp["is_dark"]
        n_dark = int((is_dark == 1.0).sum())
        n_coop = int((is_dark == 0.0).sum())
        n_unknown = int(is_dark.isna().sum())
        n_total = len(grp)
        a = float(areas.get(name, np.nan))
        rows.append({
            "mpa_name": name,
            "n_total": n_total,
            "n_dark": n_dark,
            "n_cooperative": n_coop,
            "n_unknown": n_unknown,
            "dark_ratio": n_dark / max(n_dark + n_coop, 1),
            "area_km2": a,
            "dark_density_per_km2_per_pass":
                n_dark / a / max(n_passes, 1) if a and np.isfinite(a) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_dark", ascending=False)
