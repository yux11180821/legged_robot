#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TAGS = {
    "reward": "reward",
    "distance_to_goal": "metrics/distance_to_goal",
    "success": "metrics/success",
    "spl": "metrics/spl",
}


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def read_event_scalar(event_file: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    values = accumulator.Scalars(tag)
    return (
        np.array([item.step for item in values], dtype=float),
        np.array([item.value for item in values], dtype=float),
    )


def collect(tb_root: Path, tag: str, smooth: float) -> tuple[np.ndarray, np.ndarray]:
    runs = []
    for seed_dir in sorted(tb_root.glob("seed_*")):
        event_files = sorted(seed_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            continue
        steps, values = read_event_scalar(event_files[-1], tag)
        runs.append((steps, ema(values, smooth)))
    if not runs:
        raise RuntimeError(f"No TensorBoard events found in {tb_root}")

    max_start = max(steps[0] for steps, _ in runs)
    min_end = min(steps[-1] for steps, _ in runs)
    reference = max(runs, key=lambda item: len(item[0]))[0]
    common = reference[(reference >= max_start) & (reference <= min_end)]
    if common.size == 0:
        common = np.linspace(max_start, min_end, 400)
    values = np.vstack([np.interp(common, steps, vals) for steps, vals in runs])
    return common, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smooth", type=float, default=0.85)
    parser.add_argument("--baseline-label", default="No memory")
    parser.add_argument("--rnn-label", default="RNN memory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        args.rnn_label: Path(args.rnn_root),
        args.baseline_label: Path(args.baseline_root),
    }
    colors = {
        args.rnn_label: "#2563eb",
        args.baseline_label: "#dc2626",
    }

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
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    fig.suptitle("PPO PointNav Baseline Comparison, 4 Seeds", fontsize=15, fontweight="bold")
    specs = [
        ("reward", "Reward", "Reward"),
        ("distance_to_goal", "Distance to Goal", "Meters"),
        ("success", "Success", "Rate"),
        ("spl", "SPL", "Rate"),
    ]
    summary = []
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        for label, root in roots.items():
            steps, values = collect(root, TAGS[metric], args.smooth)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            ax.plot(steps, mean, label=label, color=colors[label], linewidth=2.2)
            ax.fill_between(steps, mean - std, mean + std, color=colors[label], alpha=0.13, linewidth=0)
            summary.append(
                {
                    "metric": metric,
                    "method": label,
                    "start_mean": float(mean[0]),
                    "end_mean": float(mean[-1]),
                    "end_std": float(std[-1]),
                    "num_seeds": int(values.shape[0]),
                    "end_step": int(steps[-1]),
                }
            )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Environment frames")
    axes[0, 0].legend(frameon=False)
    fig.text(0.01, 0.01, f"Line: mean over seeds. Shading: +/- std. EMA smoothing: {args.smooth}.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    fig_path = out_dir / "ppo_pointnav_rnn_vs_no_memory.png"
    csv_path = out_dir / "ppo_pointnav_rnn_vs_no_memory_summary.csv"
    fig.savefig(fig_path, dpi=220)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(fig_path)
    print(csv_path)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
