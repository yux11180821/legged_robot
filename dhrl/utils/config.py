# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""YAML config loading.

A config may declare ``defaults: [base]`` to inherit from sibling YAML files
(e.g. ``base.yaml``); those are deep-merged first, then the config's own keys
override.  The result is a plain nested dict consumed by ``run.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge ``b`` into ``a`` (b wins); returns a new dict."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving its ``defaults`` against sibling files."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    merged: dict[str, Any] = {}
    for name in cfg.pop("defaults", []):
        base = yaml.safe_load((path.parent / f"{name}.yaml").read_text()) or {}
        merged = _deep_merge(merged, base)
    return _deep_merge(merged, cfg)
