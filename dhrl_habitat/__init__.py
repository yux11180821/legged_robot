# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""D-HRL paper reproduction in Habitat (arXiv:2407.06499), advisor-aligned.

Pipeline (see README.md):
  stage 1   lower_nav_train.py    single-agent PPO point-goal nav (one Spot
                                  learns; the other Spot is a scripted walk), frozen
  stage 2   upper_mappo_train.py  MAPPO commanding the frozen lower nav policy
                                  (dhrl / no_memory / no_hierarchy)
  plots     plot_curves.py        paper-style shadow curves + inference-time table

Pure, locally-tested building blocks: layers, marl_ppo, geometry.
Habitat-touching modules import habitat lazily (habitat_io, lower_nav_env, multi_env).
"""

from .geometry import (
    build_command,
    heading_delta_to,
    manual_position_command,
    manual_velocity_command,
    rel_target_ego,
    world_to_ego,
    wrap_angle,
)
from .layers import (
    ACTION_DIM,
    ALGORITHMS,
    COMMAND_DIM,
    EXO_OBS_DIM,
    LOWER_OBS_DIM,
    CentralCritic,
    DHRLActor,
    FlatActor,
    LowerOperator,
    build_actor,
    build_lower_obs,
    load_frozen_lower,
    save_lower_checkpoint,
)
from .marl_ppo import MARLBatch, broadcast_team_advantage, compute_gae_truncated, marl_ppo_update

__all__ = [
    "ACTION_DIM", "ALGORITHMS", "COMMAND_DIM", "EXO_OBS_DIM", "LOWER_OBS_DIM",
    "CentralCritic", "DHRLActor", "FlatActor", "LowerOperator",
    "build_actor", "build_lower_obs", "load_frozen_lower", "save_lower_checkpoint",
    "MARLBatch", "broadcast_team_advantage", "compute_gae_truncated", "marl_ppo_update",
    "world_to_ego", "heading_delta_to", "wrap_angle", "rel_target_ego",
    "manual_velocity_command", "manual_position_command", "build_command",
]
