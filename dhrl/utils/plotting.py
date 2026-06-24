# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Shadow-curve plotting for the whole project.

The CANONICAL plotter is the advisor-provided ``util.plot_learning_curve`` (mean
+/- std band over seeds) -- every figure in the project goes through it so they
all look the same.  This module only:
  (a) forces a headless matplotlib backend so it runs on the training servers, and
  (b) adapts the trainers' per-seed metric CSVs into the
      {experiment: [{steps, reward, seed}, ...]} format the plotter expects.

Do NOT add a second plotting implementation here -- use ``plot_learning_curve``.
"""
from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")  # headless: must be set BEFORE util imports pyplot

# Re-export the advisor's canonical utilities (single source of truth).
from .util import plot_learning_curve, save_results, visualize_benchmark  # noqa: E402,F401


def load_csv_curves(results_dir: str, name: str, *, metric: str = "reward") -> dict[str, list[dict]]:
    """Read ``{name}_seed_*_metrics.csv`` under ``results_dir`` into the
    {name: [{'steps', 'reward', 'seed'}, ...]} dict that ``plot_learning_curve``
    consumes.  The chosen ``metric`` column is placed under the 'reward' key
    (the plotter always reads 'reward'); seed is parsed from the filename.
    """
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{name}_seed_*_metrics.csv"))):
        base = os.path.basename(path)
        try:
            seed = int(base.split("_seed_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    steps = float(r["env_steps"])
                    value = float(r[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if steps == steps and value == value:  # drop NaN
                    rows.append({"steps": steps, "reward": value, "seed": seed})
    return {name: rows}
