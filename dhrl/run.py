# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Single entry point -- dispatches to a stage runner from the YAML config.

    python -m dhrl.run --config dhrl/configs/lower_locomotion.yaml      # stage 1
    python -m dhrl.run --config dhrl/configs/corridor_crossing.yaml     # stage 2 (IPPO)

CALL CHAIN (trace this to understand the whole project):

    run.main()
      ├─ load_config(--config)         # configs/ (base + stage/task), deep-merged
      ├─ set_seed(seed)                # utils.seeding (= util.set_seed)
      └─ dispatch on cfg['stage']:
           'lower' ─▶ runners.lower_trainer.train_lower(cfg)
                        LocomotionEnv ─▶ LowerLayer ─▶ ppo_single loop ─▶ freeze + save
           'upper' ─▶ runners.upper_trainer.train_upper(cfg)
                        HabitatBackend+task ─▶ HRLPolicy(UL+ML+frozen LL) + DecentralizedCritic
                        ─▶ IPPO + RolloutBuffer loop ─▶ save
                        (plotting via utils.plotting -> util.plot_learning_curve)
"""
from __future__ import annotations

import argparse

from dhrl.utils.config import load_config
from dhrl.utils.seeding import set_seed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Distributed HRL (IPPO) reproduction.")
    parser.add_argument("--config", required=True, help="path to a YAML config under dhrl/configs/")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg["seed"] = args.seed if args.seed is not None else cfg.get("seeds", [100])[0]
    set_seed(cfg["seed"])

    stage = cfg.get("stage")
    if stage == "lower":
        from dhrl.runners.lower_trainer import train_lower
        out = train_lower(cfg)
    elif stage == "upper":
        from dhrl.runners.upper_trainer import train_upper
        out = train_upper(cfg)
    else:
        raise SystemExit(f"config 'stage' must be 'lower' or 'upper', got {stage!r}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
