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

from dhrl.common.config import load_config
from dhrl.common.seeding import set_seed


def main(argv=None) -> None:
    """Parse args, load+seed the config, and dispatch to the stage-1 or stage-2 runner.

    Single entry point for the whole two-stage reproduction (§5.2): it reads one YAML config,
    fixes the global RNG seed for reproducibility, and routes to the lower-layer pre-training
    (``stage == 'lower'``, §5.2.1) or the upper-layer IPPO cooperation training
    (``stage == 'upper'``, §5.2.2). The stage runners are imported lazily inside each branch so
    that, e.g., a stage-1 run never has to import the heavy Habitat stack pulled in by stage 2.

    Args:
        argv (list[str] | None): CLI tokens to parse; ``None`` -> use ``sys.argv`` (the normal
            ``python -m dhrl.run ...`` path). Recognizes ``--config`` (required) and ``--seed``.

    Returns:
        None. Side effects: trains a model and prints the resulting checkpoint path to stdout.
    """
    parser = argparse.ArgumentParser(description="Distributed HRL (IPPO) reproduction.")
    parser.add_argument("--config", required=True, help="path to a YAML config under dhrl/configs/")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)                   # deep-merged base + stage/task config (configs/)
    # CLI --seed wins; else take the first entry of the config's ``seeds`` list (default [100]).
    cfg["seed"] = args.seed if args.seed is not None else cfg.get("seeds", [100])[0]
    set_seed(cfg["seed"])                             # seed python/numpy/torch RNGs for reproducibility

    stage = cfg.get("stage")
    if stage == "lower":
        # Stage 1: single-agent pre-training of the locomotion operator, then freeze + save.
        from dhrl.training.stage1_locomotion import train_lower
        out = train_lower(cfg)
    elif stage == "upper":
        # Stage 2: decentralized IPPO over UL+ML with the frozen stage-1 LL on a cooperation task.
        from dhrl.training.stage2_cooperation import train_upper
        out = train_upper(cfg)
    else:
        # Any other ``stage`` value is a config error -> abort with a clear message.
        raise SystemExit(f"config 'stage' must be 'lower' or 'upper', got {stage!r}")
    print(f"done -> {out}")                           # emit the saved checkpoint path


if __name__ == "__main__":
    main()
