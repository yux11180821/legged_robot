# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate / roll out a trained per-leg quadruped MARL policy.

Also demonstrates the UPPER-LAYER interface: with ``--external_commands`` the
velocity command (vx, vy, wz) is injected each step via
``env.unwrapped.set_velocity_commands(...)`` instead of being randomly sampled.
This is exactly how the future Habitat Commander will drive the frozen lower
layer.

    ./isaaclab.sh -p isaaclab_lowlevel/play.py --algorithm MAPPO \
        --checkpoint <path/to/agent.pt> --num_envs 16 --external_commands
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a per-leg quadruped MARL policy.")
parser.add_argument("--task", type=str, default="Isaac-Quadruped-FourLegs-Direct-MARL-v0")
parser.add_argument("--algorithm", type=str, default="MAPPO", choices=["MAPPO", "IPPO"])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default=None, help="Trained skrl checkpoint (.pt).")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument(
    "--external_commands",
    action="store_true",
    help="Drive the velocity command externally (stand-in for the upper Commander).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import isaaclab_lowlevel  # noqa: F401
from isaaclab_lowlevel.quadruped_four_legs_env import QuadrupedFourLegsEnvCfg

from isaaclab_rl.skrl import SkrlVecEnvWrapper

if args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
else:
    from skrl.utils.runner.torch import Runner


def main() -> None:
    env_cfg = QuadrupedFourLegsEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.external_commands = args_cli.external_commands  # freeze command -> upper layer drives it

    fname = "skrl_mappo_cfg.yaml" if args_cli.algorithm == "MAPPO" else "skrl_ippo_cfg.yaml"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", fname)) as f:
        agent_cfg = yaml.safe_load(f)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped  # QuadrupedFourLegsEnv (has set_velocity_commands)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    runner = Runner(env, agent_cfg)
    if args_cli.checkpoint:
        runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    # Example external command: walk forward at 0.6 m/s (upper Commander would replace this).
    fwd_cmd = torch.zeros(args_cli.num_envs, 3, device=base_env.device)
    fwd_cmd[:, 0] = 0.6

    for _ in range(args_cli.steps):
        if args_cli.external_commands:
            base_env.set_velocity_commands(fwd_cmd)
        with torch.inference_mode():
            actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
        obs, _, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
