# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Go2 per-leg MARL lower layer in MuJoCo (4 legs = 4 agents).

The wheel-legged transfer target's LOWER layer: the paper's distributed
hierarchical MARL applied at the LEG level.  Reuses the project's CTDE PPO core,
tanh action distribution and metrics harness.

Modules:
  policy        LegActor (per-leg actor) + CentralCritic (reused)   [pure torch]
  go2_leg_env   Go2LegMARLEnv (MuJoCo, 4 leg-agents)                [needs mujoco]
  train         CTDE MAPPO training loop                            [needs mujoco]
  selftest      pure-logic tests (no mujoco)

Run training on a box with mujoco + the Go2 model (see README.md):
  python mujoco_leg_marl/train.py --num-envs 32 --total-steps 3000000
"""
