"""Stage 1 — the tiling pipeline.

xView3 scenes are ~29,000 x 24,000 pixel GeoTIFFs: far too large for GPU
memory, so we cut them into overlapping tiles. The two design rules that
matter most:

1. *Windowed reads.* We never load a whole scene into RAM. rasterio reads
   only the requested pixel window straight off disk.
2. *Geo-referencing survives tiling.* Every tile remembers its (row, col)
   offset inside the parent scene, and the scene keeps its affine
   geotransform + CRS, so any pixel in any tile can be mapped back to a
   latitude/longitude. Detections are useless if you can't put them on a map.

SAR normalisation: amplitude in dB is clipped to [db_min, db_max] and
scaled to [0, 1]. Ancillary layers (bathymetry, wind) get their own clip
ranges. NaN / nodata pixels are filled with the channel minimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class TileSpec:
    """Location of one tile inside its parent scene."""

    scene_id: str
    row_off: int
    col_off: int
    height: int
    width: int


class SceneReader:
    """Windowed, normalised access to one xView3 scene folder.

    Opens the channel rasters lazily and serves normalised
    (C, H, W) float32 tiles plus the land/ice mask for any window.
    """

    def __init__(self, scene_dir: str | Path, cfg):
        self.scene_dir = Path(scene_dir)
        self.scene_id = self.scene_dir.name
        self.cfg = cfg

        self.channel_files = [self.scene_dir / name for name in cfg.scene.channels]
        self.ancillary_files = [self.scene_dir / name for name in cfg.scene.ancillary]
        self.mask_file = self.scene_dir / cfg.scene.mask_file

        # All xView3 layers are co-registered: same grid, same geotransform.
        with rasterio.open(self.channel_files[0]) as src:
            self.height = src.height
            self.width = src.width
            self.transform = src.transform
            self.crs = src.crs

        self._handles: dict[Path, rasterio.DatasetReader] = {}

    # ------------------------------------------------------------------
    # Raw I/O
    # ------------------------------------------------------------------
    def _open(self, path: Path) -> rasterio.DatasetReader:
        if path not in self._handles:
            self._handles[path] = rasterio.open(path)
        return self._handles[path]

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def _read_window(self, path: Path, spec: TileSpec) -> np.ndarray:
        """Read one band for one window; pad with NaN at scene edges."""
        src = self._open(path)
        window = Window(spec.col_off, spec.row_off, spec.width, spec.height)
        arr = src.read(
            1, window=window, boundless=True, fill_value=np.nan
        ).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_scale(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Clip to [lo, hi], scale to [0, 1], fill NaNs with 0."""
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return np.nan_to_num(arr, nan=0.0)

    def read_tile(self, spec: TileSpec) -> np.ndarray:
        """Return the normalised (C, H, W) input stack for one window.

        Channel order: VH_dB, VV_dB, bathymetry, wind speed — must match
        cfg.model.in_channels.
        """
        t = self.cfg.tiling
        bands = []
        for path in self.channel_files:                      # SAR dB channels
            bands.append(self._clip_scale(self._read_window(path, spec), t.db_min, t.db_max))
        for path in self.ancillary_files:                    # ancillary layers
            arr = self._read_window(path, spec)
            if "bathymetry" in path.name.lower():
                bands.append(self._clip_scale(arr, t.bathy_min, t.bathy_max))
            else:                                            # wind speed
                bands.append(self._clip_scale(arr, 0.0, t.wind_max))
        return np.stack(bands, axis=0)

    def read_mask(self, spec: TileSpec) -> np.ndarray:
        """Land/ice mask for the window: 0 = sea, >0 = land or ice."""
        arr = self._read_window(self.mask_file, spec)
        return np.nan_to_num(arr, nan=0.0)

    # ------------------------------------------------------------------
    # Geo-referencing
    # ------------------------------------------------------------------
    def pixel_to_lonlat(self, rows: np.ndarray, cols: np.ndarray):
        """Map scene pixel coordinates to WGS84 longitude/latitude."""
        xs, ys = rasterio.transform.xy(self.transform, rows, cols)
        xs, ys = np.asarray(xs), np.asarray(ys)
        if self.crs is not None and not self.crs.is_geographic:
            from pyproj import Transformer

            tr = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
            xs, ys = tr.transform(xs, ys)
        return xs, ys  # lon, lat


def iter_tile_specs(scene_id: str, height: int, width: int,
                    tile_size: int, overlap: int):
    """Yield TileSpecs covering the scene with the requested overlap.

    The stride is tile_size - overlap, so a vessel sitting on a tile seam
    appears fully inside at least one tile. Duplicate detections in the
    overlap strips are removed later by geographic NMS.
    """
    stride = tile_size - overlap
    n_rows = max(1, math.ceil((height - overlap) / stride))
    n_cols = max(1, math.ceil((width - overlap) / stride))
    for i in range(n_rows):
        for j in range(n_cols):
            row_off = min(i * stride, max(0, height - tile_size))
            col_off = min(j * stride, max(0, width - tile_size))
            yield TileSpec(scene_id, row_off, col_off, tile_size, tile_size)
