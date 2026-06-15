#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Paper-style shadow curves from stage CSVs (mean ± std across seeds).

Reads the metrics CSVs written by the trainers
(results/<exp>/<algorithm>_seed_<seed>_metrics.csv with columns
 env_steps, reward, ... from habitat_commander.metrics.StepLogger) and plots
one curve per algorithm with a std shadow, matching the paper's Fig.4/5 style.

    python dhrl_habitat/plot_curves.py --results-dir results/dhrl_repro \
        --algorithms dhrl no_memory no_hierarchy --metric reward

Also prints a final-window summary table (mean ± std of the last 10% of steps)
to compare against the original paper's curves, and the inference-time stats.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["orange", "forestgreen", "crimson", "dodgerblue", "darkviolet", "gold"]


def load_runs(results_dir: str, algorithm: str, metric: str):
    """-> list of (steps[np], values[np]) per seed."""
    pattern = os.path.join(results_dir, f"{algorithm}_seed_*_metrics.csv")
    runs = []
    for path in sorted(glob.glob(pattern)):
        steps, vals = [], []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    s = float(row["env_steps"])
                    v = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(s) and np.isfinite(v):
                    steps.append(s)
                    vals.append(v)
        if steps:
            runs.append((np.asarray(steps), np.asarray(vals)))
        else:
            print(f"warn: no usable rows in {path}")
    return runs


def bin_curve(runs, bin_size: float):
    """Bin each seed's curve onto a common grid -> (grid, mean, std)."""
    max_step = max(float(s.max()) for s, _ in runs)
    edges = np.arange(0.0, max_step + bin_size, bin_size)
    grid = edges[:-1] + bin_size / 2
    per_seed = []
    for steps, vals in runs:
        idx = np.clip(np.digitize(steps, edges) - 1, 0, len(grid) - 1)
        sums = np.zeros(len(grid))
        cnts = np.zeros(len(grid))
        np.add.at(sums, idx, vals)
        np.add.at(cnts, idx, 1.0)
        with np.errstate(invalid="ignore"):
            curve = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        # forward-fill empty bins
        last = np.nan
        for i in range(len(curve)):
            if np.isnan(curve[i]):
                curve[i] = last
            else:
                last = curve[i]
        per_seed.append(curve)
    stack = np.vstack(per_seed)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    ok = ~np.isnan(mean)
    return grid[ok], mean[ok], std[ok]


def main() -> None:
    p = argparse.ArgumentParser(description="Plot D-HRL reproduction shadow curves.")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--algorithms", nargs="+", default=["dhrl", "no_memory", "no_hierarchy"])
    p.add_argument("--metric", default="reward")
    p.add_argument("--bin-size", type=float, default=None,
                   help="Step-bin width; default = max_steps/100.")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default="D-HRL Reproduction (Habitat)")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))
    summary: dict[str, str] = {}
    infer_summary: dict[str, str] = {}

    for i, algo in enumerate(args.algorithms):
        runs = load_runs(args.results_dir, algo, args.metric)
        if not runs:
            print(f"warn: no runs found for algorithm '{algo}' in {args.results_dir}")
            continue
        max_step = max(float(s.max()) for s, _ in runs)
        bin_size = args.bin_size or max(1.0, max_step / 100.0)
        grid, mean, std = bin_curve(runs, bin_size)
        color = COLORS[i % len(COLORS)]
        ax.plot(grid, mean, color=color, linewidth=2.2, label=f"{algo} ({len(runs)} seeds)")
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)

        tail = grid >= 0.9 * grid.max()
        summary[algo] = f"{np.mean(mean[tail]):.3f} ± {np.mean(std[tail]):.3f}"

        # inference-time summary (the reviewer-rebuttal metric)
        inf_runs = load_runs(args.results_dir, algo, "infer_ms_mean")
        if inf_runs:
            finals = [v[-1] for _, v in inf_runs]
            infer_summary[algo] = f"{np.mean(finals):.3f} ms/step"

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel(args.metric, fontsize=13)
    ax.set_title(args.title, fontsize=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = args.out or os.path.join(args.results_dir, f"curves_{args.metric}.png")
    fig.savefig(out, dpi=300)
    print(f"saved {out}\n")
    print(f"final-window (last 10% of steps) {args.metric}:")
    for algo, txt in summary.items():
        print(f"  {algo:14s} {txt}")
    if infer_summary:
        print("single-step inference time (推理一步耗时):")
        for algo, txt in infer_summary.items():
            print(f"  {algo:14s} {txt}")


if __name__ == "__main__":
    main()
