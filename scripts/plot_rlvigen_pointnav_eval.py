#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["success", "spl", "reward", "distance_to_goal"]
CONDITION_LABELS = {
    "default": "Default",
    "fov_60": "FOV 60",
    "fov_120": "FOV 120",
    "camera_low": "Camera Low",
    "camera_high": "Camera High",
}


def parse_log(path: Path) -> dict[str, float]:
    out = {}
    pattern = re.compile(r"Average episode ([^:]+):\s+([-+0-9.eE]+)")
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            out[match.group(1)] = float(match.group(2))
    missing = [metric for metric in METRICS if metric not in out]
    if missing:
        raise ValueError(f"{path} missing metrics: {missing}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="RL-ViGen-style PointNav Camera Generalization")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log in sorted(log_dir.glob("*_seed_*.log")):
        stem = log.stem
        cond, seed_part = stem.rsplit("_seed_", 1)
        metrics = parse_log(log)
        rows.append(
            {
                "condition": cond,
                "seed": int(seed_part),
                **{metric: metrics[metric] for metric in METRICS},
            }
        )

    if not rows:
        raise RuntimeError(f"No eval logs found in {log_dir}")

    raw_csv = out_dir / "rlvigen_pointnav_eval_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    conditions = [c for c in CONDITION_LABELS if any(r["condition"] == c for r in rows)]
    summary = []
    for cond in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond]
        for metric in METRICS:
            vals = np.array([r[metric] for r in cond_rows], dtype=float)
            summary.append(
                {
                    "condition": cond,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "num_seeds": int(vals.size),
                }
            )

    summary_csv = out_dir / "rlvigen_pointnav_eval_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "metric", "mean", "std", "num_seeds"])
        writer.writeheader()
        writer.writerows(summary)

    def metric_stats(metric: str) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for cond in conditions:
            vals = np.array([r[metric] for r in rows if r["condition"] == cond], dtype=float)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        return np.array(means), np.array(stds)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
    fig.suptitle(args.title, fontsize=15, fontweight="bold")
    colors = {
        "success": "#2563eb",
        "spl": "#16a34a",
        "reward": "#9333ea",
        "distance_to_goal": "#dc2626",
    }
    titles = {
        "success": "Success",
        "spl": "SPL",
        "reward": "Reward",
        "distance_to_goal": "Distance to Goal",
    }
    ylabels = {
        "success": "Rate",
        "spl": "Rate",
        "reward": "Return",
        "distance_to_goal": "Meters",
    }
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    x = np.arange(len(conditions))
    for ax, metric in zip(axes.flat, METRICS):
        means, stds = metric_stats(metric)
        ax.bar(x, means, yerr=stds, color=colors[metric], alpha=0.82, capsize=4)
        ax.set_title(titles[metric], loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabels[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(0.01, 0.01, "Bars: mean over 4 seeds. Error bars: +/- std. Evaluation: 50 val episodes per seed.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig_path = out_dir / "rlvigen_pointnav_camera_generalization.png"
    fig.savefig(fig_path, dpi=220)

    print(fig_path)
    print(summary_csv)
    print(raw_csv)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
