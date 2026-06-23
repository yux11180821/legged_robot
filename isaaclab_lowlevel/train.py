# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone skrl MAPPO/IPPO trainer for the per-leg quadruped MARL env.

Self-contained (no Isaac Lab hydra task registry needed): it registers the env
by importing this package, builds the env cfg directly, loads the skrl yaml, and
runs the skrl Runner. Run with Isaac Lab's python:

    ./isaaclab.sh -p isaaclab_lowlevel/train.py --algorithm MAPPO --num_envs 4096 --headless
    ./isaaclab.sh -p isaaclab_lowlevel/train.py --algorithm IPPO  --num_envs 4096 --headless --max_iterations 1500

(From the repo root, with this repo on PYTHONPATH.)
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------- #
# 1. CLI + AppLauncher  (MUST come before importing isaaclab.envs / the env).
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Train per-leg quadruped MARL with skrl.")
parser.add_argument("--task", type=str, default="Isaac-Quadruped-FourLegs-Direct-MARL-v0")
parser.add_argument("--algorithm", type=str, default="MAPPO", choices=["MAPPO", "IPPO"])
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a checkpoint.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# 2. Post-launch imports (need the running sim app).
# --------------------------------------------------------------------------- #
import os
import sys

import gymnasium as gym
import yaml

# Make this repo importable, then import the package to register the gym id.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import isaaclab_lowlevel  # noqa: F401  -> registers Isaac-Quadruped-FourLegs-Direct-MARL-v0
from isaaclab_lowlevel.quadruped_four_legs_env import QuadrupedFourLegsEnvCfg

from isaaclab_rl.skrl import SkrlVecEnvWrapper

if args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
else:
    from skrl.utils.runner.torch import Runner


def _agent_cfg_path() -> str:
    fname = "skrl_mappo_cfg.yaml" if args_cli.algorithm == "MAPPO" else "skrl_ippo_cfg.yaml"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", fname)


def main() -> None:
    # --- env cfg ---
    env_cfg = QuadrupedFourLegsEnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # --- agent cfg (skrl yaml) ---
    with open(_agent_cfg_path()) as f:
        agent_cfg = yaml.safe_load(f)
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]
    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]

    # --- env ---  (DirectMARLEnv is preserved for MAPPO/IPPO: do NOT flatten) ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # --- skrl runner builds per-agent actors + the (centralized) critic ---
    runner = Runner(env, agent_cfg)
    if args_cli.checkpoint:
        runner.agent.load(args_cli.checkpoint)

    runner.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
