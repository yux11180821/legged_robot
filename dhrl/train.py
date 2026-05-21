from __future__ import annotations

import argparse
from dataclasses import replace

from .config import load_config
from .ippo import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train D-HRL/IPPO reproduction on proxy embodied-cooperation tasks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=["dhrl", "no_memory", "no_hierarchy"], default="dhrl")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    updates = {}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.total_steps is not None:
        updates["total_steps"] = args.total_steps
    if args.num_envs is not None:
        updates["num_envs"] = args.num_envs
    config = replace(config, **updates)
    train(config, args.variant, device_name=args.device)


if __name__ == "__main__":
    main()
