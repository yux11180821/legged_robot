#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Paper-style shadow curves for the D-HRL reproduction.

Uses the user's OWN shadow plotter (scripts/user_shadow_util.py
:func:`plot_learning_curve` -- mean +/- std band + per-seed faint traces +
normalized paper-style x-axis) so these figures match every other figure in the
project.  It just adapts the dhrl_habitat metrics CSVs (written every --log-every
env steps by habitat_commander.metrics.StepLogger) into that function's
``experiments_data`` format.

    python dhrl_habitat/plot_curves.py --results-dir results/dhrl_repro \
        --algorithms dhrl no_memory no_hierarchy --metric reward
    python dhrl_habitat/plot_curves.py --results-dir results/dhrl_repro \
        --algorithms dhrl no_memory no_hierarchy --metric success_rate

Also prints the single-step inference-time table (推理一步耗时) -- the efficiency
metric for the reviewer rebuttal.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

# Use the project's own shadow-plot utility (keeps figure style consistent).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from user_shadow_util import plot_learning_curve  # noqa: E402

# Paper labels (Fig.5 / Table 1).
LABELS = {
    "dhrl": "Distributed HRL",
    "no_memory": "No Spatiotemporal Memory",
    "no_hierarchy": "No Hierarchy",
}


def load_rows(results_dir: str, algorithm: str, metric: str) -> list[dict]:
    """metrics CSV(s) -> [{'steps', 'reward', 'seed'}] for plot_learning_curve.

    The metric's value is placed under the 'reward' key because
    plot_learning_curve always reads that key (its ylabel arg sets the display
    label).  seed is parsed from the filename.
    """
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{algorithm}_seed_*_metrics.csv"))):
        base = os.path.basename(path)
        try:
            seed = int(base.split("_seed_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    steps = float(row["env_steps"])
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(steps) and np.isfinite(value):
                    rows.append({"steps": steps, "reward": value, "seed": seed})
    return rows


def inference_table(results_dir: str, algorithm: str) -> str | None:
    """Final-step single-step inference time averaged over seeds (ms)."""
    finals = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{algorithm}_seed_*_metrics.csv"))):
        last = None
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    last = float(row["infer_ms_mean"])
                except (KeyError, TypeError, ValueError):
                    continue
        if last is not None:
            finals.append(last)
    if not finals:
        return None
    return f"{np.mean(finals):.3f} ms/step  (seeds={len(finals)})"


def main() -> None:
    p = argparse.ArgumentParser(description="D-HRL reproduction shadow curves (user style).")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--algorithms", nargs="+", default=["dhrl", "no_memory", "no_hierarchy"])
    p.add_argument("--metric", default="reward", help="CSV column to plot (reward / success_rate / ...).")
    p.add_argument("--bin-size", type=int, default=None, help="Step-bin width; default = max_step/80.")
    p.add_argument("--raw-x", action="store_true", help="Raw env steps instead of normalized [0,1] x-axis.")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    experiments_data: dict[str, list[dict]] = {}
    observed_max = 0.0
    for algo in args.algorithms:
        rows = load_rows(args.results_dir, algo, args.metric)
        if not rows:
            print(f"warn: no rows for '{algo}' (metric={args.metric}) in {args.results_dir}")
            continue
        experiments_data[LABELS.get(algo, algo)] = rows
        observed_max = max(observed_max, max(r["steps"] for r in rows))

    if not experiments_data:
        raise SystemExit("no data to plot")

    ylabel = "Episode Reward" if args.metric == "reward" else args.metric.replace("_", " ").title()
    title = args.title or f"D-HRL Reproduction (Habitat, ReplicaCAD) — {args.metric}"
    out = args.out or os.path.join(args.results_dir, f"dhrl_{args.metric}_shadow.png")
    bin_size = args.bin_size or max(1, int(observed_max / 80))

    plot_learning_curve(
        experiments_data=experiments_data,
        bin_size=bin_size,
        title=title,
        output_filename=out,
        ylabel=ylabel,
        normalize_x=not args.raw_x,
        x_max=observed_max,
        plot_seed_lines=True,
        show=False,
    )

    print(f"\nsingle-step inference time (推理一步耗时):")
    for algo in args.algorithms:
        t = inference_table(args.results_dir, algo)
        if t:
            print(f"  {LABELS.get(algo, algo):28s} {t}")


if __name__ == "__main__":
    main()
