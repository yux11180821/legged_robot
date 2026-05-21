#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(log_dir: Path, metric: str) -> dict[str, dict[int, list[tuple[int, float]]]]:
    data: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for path in log_dir.glob("*/seed_*/metrics.csv"):
        variant = path.parent.parent.name
        seed = int(path.parent.name.replace("seed_", ""))
        data.setdefault(variant, {}).setdefault(seed, [])
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                value = row.get(metric, "")
                if value == "":
                    continue
                data[variant][seed].append((int(row["step"]), float(value)))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--metric", default="success_rate")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="D-HRL Embodied Cooperation")
    args = parser.parse_args()

    data = load_rows(Path(args.log_dir), args.metric)
    if not data:
        raise SystemExit(f"no metrics found under {args.log_dir}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"dhrl": "#2563eb", "no_memory": "#f59e0b", "no_hierarchy": "#64748b"}

    for variant, seed_map in sorted(data.items()):
        common_steps = sorted(set.intersection(*(set(s for s, _ in rows) for rows in seed_map.values())))
        if not common_steps:
            continue
        matrix = []
        for seed, rows in sorted(seed_map.items()):
            row_dict = dict(rows)
            values = np.array([row_dict[s] for s in common_steps], dtype=np.float32)
            matrix.append(values)
            ax.plot(common_steps, values, color=colors.get(variant, None), alpha=0.18, linewidth=1.0)
        arr = np.stack(matrix, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = colors.get(variant, None)
        ax.plot(common_steps, mean, label=f"{variant} ({len(matrix)} seeds)", color=color, linewidth=2.5)
        ax.fill_between(common_steps, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_title(args.title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(args.metric)
    ax.legend()
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
