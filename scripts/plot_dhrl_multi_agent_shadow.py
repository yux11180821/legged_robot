#!/usr/bin/env python3
"""Plot two-algorithm D-HRL shadow curves using the user's utility function."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


TAGS = {
    "reward": "reward",
    "success": "metrics/social_nav_seek_success",
    "pddl_success": "metrics/pddl_success",
    "spl": "metrics/spl",
}


DEFAULT_ALGORITHMS = {
    "dhrl": "Distributed HRL",
    "no_memory": "No Spatiotemporal Memory",
}


def parse_algorithm(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        return raw, DEFAULT_ALGORITHMS.get(raw, raw)
    key, label = raw.split("=", 1)
    return key.strip(), label.strip()


def read_scalar(event_file: Path, tag: str) -> list[dict[str, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard is required to read training curves. "
            "Install it in the Habitat environment or run this script inside the "
            "AutoDL conda env where habitat-baselines writes event files."
        ) from exc

    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    if tag not in available:
        raise KeyError(
            f"{event_file} does not contain scalar '{tag}'. "
            f"Available examples: {sorted(available)[:20]}"
        )
    return [
        {"steps": int(item.step), "reward": float(item.value)}
        for item in accumulator.Scalars(tag)
    ]


def collect_algorithm(tb_root: Path, algorithm: str, tag: str) -> list[dict[str, float]]:
    algo_root = tb_root / algorithm
    rows: list[dict[str, float]] = []
    for seed_dir in sorted(algo_root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        event_files = sorted(seed_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            continue
        for row in read_scalar(event_files[-1], tag):
            row["seed"] = seed
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No TensorBoard scalar data found under {algo_root}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tb-root",
        default="tb/dhrl_social_nav_2algorithms",
        help="TensorBoard root containing {algorithm}/seed_{seed}/ event files.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/dhrl_social_nav_2algorithms",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["dhrl=Distributed HRL", "no_memory=No Spatiotemporal Memory"],
        help="Algorithm keys or key=label pairs.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(TAGS),
        default="reward",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Override TensorBoard scalar tag. Defaults are selected from --metric.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--title",
        default=None,
    )
    parser.add_argument(
        "--output-name",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from user_shadow_util import plot_learning_curve, save_results

    tb_root = Path(args.tb_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or TAGS[args.metric]
    algorithms = [parse_algorithm(item) for item in args.algorithms]
    experiments_data = {}

    for algorithm, label in algorithms:
        rows = collect_algorithm(tb_root, algorithm, tag)
        experiments_data[label] = rows
        save_results(rows, out_dir / f"{algorithm}_{args.metric}_train_data.pkl")
        print(f"loaded {len(rows)} points for {label}")

    title = args.title or f"D-HRL Multi-Agent Comparison ({args.metric})"
    output_name = args.output_name or f"dhrl_multi_agent_{args.metric}_shadow.png"
    ylabel = "Episode Reward" if args.metric == "reward" else args.metric.replace("_", " ").title()

    plot_learning_curve(
        experiments_data=experiments_data,
        bin_size=args.bin_size,
        title=title,
        output_filename=str(out_dir / output_name),
        ylabel=ylabel,
        show=False,
    )


if __name__ == "__main__":
    main()
