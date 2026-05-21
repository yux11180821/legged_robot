#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path: Path) -> dict[str, float]:
    pattern = re.compile(r"Average episode ([^:]+):\s+([-+0-9.eE]+)")
    out = {}
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            out[match.group(1)] = float(match.group(2))
    if "success" not in out:
        raise ValueError(f"{path} missing success metric")
    return out


def load_step(checkpoint: Path) -> int:
    import torch

    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(obj.get("extra_state", {}).get("step", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--total-steps", type=float, required=True)
    parser.add_argument("--title", default="Zero-shot PointNav")
    parser.add_argument("--metric", default="success")
    parser.add_argument("--ylabel", default="Success Rate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 200, 300, 400])
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
        metrics = parse_log(log)
        if args.metric not in metrics:
            raise ValueError(f"{log} missing metric {args.metric!r}")
        rows.append(
            {
                "checkpoint": label,
                "seed": seed,
                "step": load_step(ckpt),
                args.metric: metrics[args.metric],
            }
        )

    deduped = {}
    for row in rows:
        key = (row["seed"], row["step"])
        if key not in deduped or row["checkpoint"] == "final":
            deduped[key] = row
    rows = sorted(deduped.values(), key=lambda row: (row["step"], row["seed"]))

    raw_csv = out_dir / f"zero_shot_{args.metric}_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "seed", "step", args.metric])
        writer.writeheader()
        writer.writerows(rows)

    steps = sorted({row["step"] for row in rows})
    xs = np.array([step / args.total_steps for step in steps], dtype=float)
    means, stds = [], []
    summary = []
    for step in steps:
        vals = np.array([row[args.metric] for row in rows if row["step"] == step], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        means.append(mean)
        stds.append(std)
        summary.append(
            {
                "step": step,
                "normalized_step": step / args.total_steps,
                "metric": args.metric,
                "mean": mean,
                "std": std,
                "num_seeds": int(vals.size),
            }
        )

    summary_csv = out_dir / f"zero_shot_{args.metric}_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "normalized_step", "metric", "mean", "std", "num_seeds"],
        )
        writer.writeheader()
        writer.writerows(summary)

    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    color = "#2563eb"
    for seed in args.seeds:
        seed_rows = sorted([row for row in rows if row["seed"] == seed], key=lambda row: row["step"])
        if not seed_rows:
            continue
        ax.plot(
            [row["step"] / args.total_steps for row in seed_rows],
            [row[args.metric] for row in seed_rows],
            color=color,
            alpha=0.18,
            linewidth=0.9,
        )
    ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.16, linewidth=0)
    ax.plot(xs, means, color=color, linewidth=2.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Normalized training steps")
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Faint lines: individual seeds. Bold line: mean. Shading: +/- std.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = out_dir / f"zero_shot_{args.metric}_paper_curve.png"
    fig.savefig(out, dpi=240)
    print(out)
    print(summary_csv)
    print(raw_csv)


if __name__ == "__main__":
    main()
