"""Configuration loading.

A single YAML file (configs/default.yaml) drives every script in the
project. This module loads it into a lightweight dot-accessible object so
code can write `cfg.tiling.tile_size` instead of `cfg["tiling"]["tile_size"]`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


class DotDict(dict):
    """A dict whose keys are also attributes, applied recursively."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def to_plain(self) -> dict:
        """Recursively convert back to built-in dicts (e.g. for pickling)."""

        def unwrap(obj):
            if isinstance(obj, dict):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            return obj

        return unwrap(self)


def load_config(path: str | Path = "configs/default.yaml") -> DotDict:
    """Load the YAML config and resolve all paths relative to the repo root."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = DotDict.wrap(copy.deepcopy(raw))

    # Resolve path entries relative to the repository root (the directory
    # that contains the configs/ folder), so scripts work from anywhere.
    repo_root = path.resolve().parent.parent

    def resolve(value):
        if isinstance(value, list):
            return [resolve(v) for v in value]
        p = Path(value)
        return str((repo_root / p).resolve()) if not p.is_absolute() else value

    for key, value in cfg.paths.items():
        cfg.paths[key] = resolve(value)
    return cfg
