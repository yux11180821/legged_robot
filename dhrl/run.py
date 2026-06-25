# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Single entry point.  Run from inside dhrl/:

    python run.py --config configs/lower_locomotion.yaml      # stage 1 (pre-train lower)
    python run.py --config configs/corridor_crossing.yaml     # stage 2 (IPPO)

Call chain (read top-to-bottom):
    main()
      get_config()                      # configs/<file>.yaml -> args Namespace
      set_seed(args.seed)
      stage == 'lower':  train_lower(args)
      stage == 'upper':  build E envs -> agent = load_algorithm(args.algo)(env_info, args)
                         -> RolloutBuffer -> train(args, envs, agent, buffer)
"""
import numpy as np

from utils.config import get_config, load_algorithm
from utils.seeding import set_seed

TASKS = {"corridor_crossing": "envs.corridor_crossing.CorridorCrossing"}


def main():
    args = get_config()
    set_seed(args.seed)

    if args.stage == "lower":
        from runners.lower_trainer import train_lower
        out = train_lower(args)

    elif args.stage == "upper":
        from envs.habitat_backend import HabitatBackend
        from memories.rollout_buffer import RolloutBuffer
        from runners.upper_trainer import train

        Task = load_algorithm(TASKS[args.task])          # the cooperation task class
        n_envs = getattr(args, "num_envs", 1)
        envs = []
        for i in range(n_envs):
            world = HabitatBackend(config_name=args.config_name, seed=args.seed + i,
                                   n_agents=getattr(args, "n_agents", 2))
            envs.append(Task(world, proprio_dim=world.proprio_dim,
                             command_dim=getattr(args, "command_dim", 3),
                             action_dim=world.action_dim, rng=np.random.default_rng(args.seed + i)))

        env_info = envs[0].get_env_info()
        agent = load_algorithm(args.algo)(env_info, args)    # e.g. algorithms.ippo.IPPO
        buffer = RolloutBuffer(args.rollout_steps, n_envs, env_info["n_agents"],
                               env_info["exo_dim"], env_info["command_dim"],
                               args.hidden_dim, args.device, use_memory=getattr(args, "use_memory", True))
        out = train(args, envs, agent, buffer)

    else:
        raise SystemExit(f"config 'stage' must be 'lower' or 'upper', got {args.stage!r}")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
