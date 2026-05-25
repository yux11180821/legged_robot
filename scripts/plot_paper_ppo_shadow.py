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
    if alpha <= 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def read_scalar(event_file: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    scalars = accumulator.Scalars(tag)
    if not scalars:
        raise ValueError(f"No scalar tag {tag!r} found in {event_file}")
    steps = np.array([item.step for item in scalars], dtype=float)
    values = np.array([item.value for item in scalars], dtype=float)
    return steps, values


def collect_runs(tb_root: Path, tag: str, smooth: float) -> tuple[np.ndarray, np.ndarray]:
    runs = []
    for seed_dir in sorted(tb_root.glob("seed_*")):
        event_files = sorted(seed_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            continue
        steps, values = read_scalar(event_files[-1], tag)
        runs.append((seed_dir.name, steps, ema(values, smooth)))

    if len(runs) < 2:
        raise RuntimeError(f"Need at least 2 seed runs under {tb_root}, found {len(runs)}")

    max_start = max(steps[0] for _, steps, _ in runs)
    min_end = min(steps[-1] for _, steps, _ in runs)
    reference = max(runs, key=lambda item: len(item[1]))[1]
    common_steps = reference[(reference >= max_start) & (reference <= min_end)]
    if common_steps.size == 0:
        common_steps = np.linspace(max_start, min_end, 400)

    values_by_seed = []
    for _, steps, values in runs:
        values_by_seed.append(np.interp(common_steps, steps, values))

    return common_steps, np.vstack(values_by_seed)


def plot_metric(ax, steps, values, title, ylabel, color, spread_mode):
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    spread = std / np.sqrt(values.shape[0]) if spread_mode == "sem" else std
    ax.plot(steps, mean, color=color, linewidth=2.3)
    ax.fill_between(steps, mean - spread, mean + spread, color=color, alpha=0.18, linewidth=0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return mean, spread


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smooth", type=float, default=0.85)
    parser.add_argument("--shade", choices=["std", "sem"], default="std")
    parser.add_argument(
        "--title",
        default="PPO PointNav End-to-End Training, 4 Random Seeds",
    )
    args = parser.parse_args()

    tb_root = Path(args.tb_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, tag in TAGS.items():
        steps, values = collect_runs(tb_root, tag, args.smooth)
        data[name] = (steps, values)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    fig.suptitle(args.title, fontsize=15, fontweight="bold")

    specs = [
        ("reward", "Reward", "Reward", "#2563eb"),
        ("distance_to_goal", "Distance to Goal", "Meters", "#dc2626"),
        ("success", "Success", "Rate", "#16a34a"),
        ("spl", "SPL", "Rate", "#9333ea"),
    ]
    summary_rows = []
    for ax, (name, title, ylabel, color) in zip(axes.flat, specs):
        steps, raw_values = data[name]
        mean, spread = plot_metric(ax, steps, raw_values, title, ylabel, color, args.shade)
        summary_rows.append(
            {
                "metric": name,
                "start_mean": float(mean[0]),
                "end_mean": float(mean[-1]),
                "end_spread": float(spread[-1]),
                "num_seeds": int(raw_values.shape[0]),
                "end_step": int(steps[-1]),
            }
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Environment frames")

    note = f"Line: mean across seeds. Shading: +/- {args.shade}. EMA smoothing: {args.smooth}."
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    fig_path = out_dir / "ppo_pointnav_4seed_shadow.png"
    fig.savefig(fig_path, dpi=220)

    csv_path = out_dir / "ppo_pointnav_4seed_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    reward_steps, reward_values = data["reward"]
    curve_csv = out_dir / "ppo_pointnav_4seed_reward_mean_std.csv"
    with curve_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean", "std"])
        reward_mean = reward_values.mean(axis=0)
        reward_std = reward_values.std(axis=0, ddof=1)
        for step, mean, std in zip(reward_steps, reward_mean, reward_std):
            writer.writerow([int(step), float(mean), float(std)])

    print(fig_path)
    print(csv_path)
    print(curve_csv)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
