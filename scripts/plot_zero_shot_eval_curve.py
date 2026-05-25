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


def parse_log(path: Path) -> dict[str, float]:
    pattern = re.compile(r"Average episode ([^:]+):\s+([-+0-9.eE]+)")
    out = {}
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            out[match.group(1)] = float(match.group(2))
    missing = [metric for metric in METRICS if metric not in out]
    if missing:
        raise ValueError(f"{path} missing metrics: {missing}")
    return {metric: out[metric] for metric in METRICS}


def load_step(checkpoint: Path) -> int:
    import torch

    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(obj.get("extra_state", {}).get("step", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 200, 300, 400])
    parser.add_argument("--title", default="Zero-shot PointNav Evaluation on Held-out Scene")
    parser.add_argument(
        "--note",
        default="Faint lines: individual seeds. Bold line: mean. Shading: +/- std.",
    )
    parser.add_argument("--hide-seed-lines", action="store_true")
    parser.add_argument(
        "--x-axis",
        choices=["frames", "normalized"],
        default="frames",
        help="Use raw training frames or normalize steps to [0, 1].",
    )
    parser.add_argument(
        "--total-steps",
        type=float,
        default=None,
        help="Denominator for normalized x-axis. Defaults to max observed step.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    ckpt_root = Path(args.checkpoint_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log in sorted(log_dir.glob("*_seed_*.log")):
        label, seed_part = log.stem.rsplit("_seed_", 1)
        seed = int(seed_part)
        ckpt = ckpt_root / f"seed_{seed}" / f"{label}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        rows.append(
            {
                "checkpoint": label,
                "seed": seed,
                "step": load_step(ckpt),
                **parse_log(log),
            }
        )

    deduped = {}
    for row in rows:
        key = (row["seed"], row["step"])
        if key not in deduped or row["checkpoint"] == "final":
            deduped[key] = row
    rows = sorted(deduped.values(), key=lambda row: (row["step"], row["seed"]))

    raw_csv = out_dir / "zero_shot_eval_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "seed", "step", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    steps = sorted({row["step"] for row in rows})
    x_denominator = args.total_steps or max(steps)

    def x_value(step: int) -> float:
        if args.x_axis == "normalized":
            return float(step) / float(x_denominator)
        return float(step)

    summary = []
    for step in steps:
        step_rows = [row for row in rows if row["step"] == step]
        for metric in METRICS:
            vals = np.array([row[metric] for row in step_rows], dtype=float)
            summary.append(
                {
                    "step": step,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "num_seeds": int(vals.size),
                }
            )

    summary_csv = out_dir / "zero_shot_eval_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "metric", "mean", "std", "num_seeds"])
        writer.writeheader()
        writer.writerows(summary)

    def metric_stats(metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, means, stds = [], [], []
        for step in steps:
            vals = np.array([row[metric] for row in rows if row["step"] == step], dtype=float)
            xs.append(x_value(step))
            means.append(vals.mean())
            stds.append(vals.std(ddof=1) if vals.size > 1 else 0.0)
        return np.array(xs), np.array(means), np.array(stds)

    def seed_stats(metric: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        seed_rows = sorted(
            [row for row in rows if row["seed"] == seed],
            key=lambda row: row["step"],
        )
        return (
            np.array([x_value(row["step"]) for row in seed_rows], dtype=float),
            np.array([row[metric] for row in seed_rows], dtype=float),
        )

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
    fig.suptitle(args.title, fontsize=15, fontweight="bold")
    colors = {
        "success": "#2563eb",
        "spl": "#16a34a",
        "reward": "#9333ea",
        "distance_to_goal": "#dc2626",
    }
    labels = {
        "success": ("Success", "Rate"),
        "spl": ("SPL", "Rate"),
        "reward": ("Reward", "Return"),
        "distance_to_goal": ("Distance to Goal", "Meters"),
    }

    for ax, metric in zip(axes.flat, METRICS):
        xs, mean, std = metric_stats(metric)
        if not args.hide_seed_lines:
            for seed in sorted({row["seed"] for row in rows}):
                seed_xs, seed_vals = seed_stats(metric, seed)
                ax.plot(
                    seed_xs,
                    seed_vals,
                    color=colors[metric],
                    linewidth=0.9,
                    alpha=0.24,
                )
        ax.plot(xs, mean, color=colors[metric], linewidth=2.3)
        ax.fill_between(xs, mean - std, mean + std, color=colors[metric], alpha=0.18, linewidth=0)
        ax.set_title(labels[metric][0], loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(labels[metric][1])
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1, :]:
        if args.x_axis == "normalized":
            ax.set_xlabel("Normalized training steps")
            ax.set_xlim(0.0, 1.0)
        else:
            ax.set_xlabel("Training frames")
    fig.text(0.01, 0.01, args.note, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    fig_path = out_dir / "zero_shot_eval_shadow_curve.png"
    fig.savefig(fig_path, dpi=220)
    print(fig_path)
    print(summary_csv)
    print(raw_csv)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
