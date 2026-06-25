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

Context for this reproduction of arXiv:2407.06499 ("Learning a Distributed
Hierarchical Locomotion Controller for Embodied Cooperation", CoRL 2024): the
paper reports its results as across-seed learning curves with a variance band
(e.g. the IPPO-vs-MAPPO/CTDE comparisons of Fig.5). To keep every figure visually
identical, ALL plotting is delegated to the advisor's verbatim
``util.plot_learning_curve``; ``common.util`` is intentionally left untouched.
This wrapper exists only so the rest of the codebase can import one stable
plotting surface that (a) is import-safe on headless training servers and
(b) reshapes the per-seed CSV traces from ``common.logging.StepLogger`` into the
plotter's expected schema. It deliberately contains NO plotting logic of its own.
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

    This is the sole adapter between the trainers' on-disk metric traces (written
    by ``common.logging.StepLogger``, one CSV per seed) and the advisor's
    canonical mean +/- std shadow-curve plotter. Pooling every seed's rows into a
    single flat list under one experiment ``name`` is exactly the format the
    plotter needs to draw the across-seed variance band -- which is how each
    §5.1 cooperation task's learning curve (and each baseline comparison, cf.
    Fig.5) is presented in this reproduction.

    Args:
        results_dir: Directory holding the per-seed metric CSVs. ``str`` path;
            globbed (not recursed) for the ``name``-matching files.
        name: Experiment identifier and filename stem; files matching
            ``{name}_seed_*_metrics.csv`` are gathered, and ``name`` is also the
            single key under which all their rows are returned (so it becomes the
            curve's label in the plot legend).
        metric: Name of the CSV column to plot on the y-axis (e.g. ``"reward"``,
            ``"success_rate"``, or an ``infer_ms_*`` column). Defaults to
            ``"reward"``. Its values are stored under the ``'reward'`` key
            regardless, because the downstream plotter always reads ``'reward'``.

    Returns:
        A dict[str, list[dict]] with a single key ``name`` mapping to a flat list
        pooling all seeds' points. Each point is a dict with: ``steps`` (float,
        the x-axis env-step count), ``reward`` (float, the chosen ``metric``'s
        value), and ``seed`` (int, parsed from the filename so the plotter can
        group/aggregate by seed). Rows whose ``env_steps`` or ``metric`` are
        missing, non-numeric, or NaN are silently skipped; files whose seed
        cannot be parsed are skipped entirely.
    """
    rows: list[dict] = []
    # `sorted` -> deterministic file order so the pooled list is reproducible run-to-run.
    for path in sorted(glob.glob(os.path.join(results_dir, f"{name}_seed_*_metrics.csv"))):
        base = os.path.basename(path)
        try:
            # Recover the integer seed from "<name>_seed_<k>_metrics.csv".
            seed = int(base.split("_seed_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue  # filename did not carry a parseable seed -> skip this file
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):  # one dict per CSV row (header-keyed)
                try:
                    steps = float(r["env_steps"])  # x-axis: cumulative env steps
                    value = float(r[metric])       # y-axis: the requested metric
                except (KeyError, TypeError, ValueError):
                    continue  # missing column or non-numeric cell -> drop this row
                # NaN != NaN, so `x == x` is a cheap NaN filter for both fields.
                if steps == steps and value == value:  # drop NaN
                    # Note: `metric`'s value is stored under 'reward' because the
                    # canonical plotter always reads the 'reward' key.
                    rows.append({"steps": steps, "reward": value, "seed": seed})
    return {name: rows}
