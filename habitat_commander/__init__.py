# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Upper-layer Commander (Habitat side) of the hybrid hierarchy.

Pure, importable building blocks (no Habitat import at module load):
  CommanderPolicy / TanhNormal / DepthCNN  -- perception -> (vx, vy, wz)
  compute_gae_truncated / ppo_update       -- bug-fixed PPO core
  VLMTeacher / OracleGoalTeacher / behavior_clone -- VLM offline distillation
  CommanderObs / CommanderReward / make_commander_env -- Habitat adapter (lazy habitat import)

Run training with:  python habitat_commander/train.py ...   (see README.md)
"""

from .commander_policy import CommanderPolicy, DepthCNN, TanhNormal
from .habitat_env import CommanderObs, CommanderReward, make_commander_env
from .metrics import InferenceTimer, StepLogger
from .ppo import PPOBatch, compute_gae_truncated, normalize_advantages, ppo_update
from .vlm_distill import ManualCommandPolicy, OracleGoalTeacher, VLMTeacher, behavior_clone

__all__ = [
    "CommanderPolicy", "DepthCNN", "TanhNormal",
    "compute_gae_truncated", "ppo_update", "normalize_advantages", "PPOBatch",
    "ManualCommandPolicy", "VLMTeacher", "OracleGoalTeacher", "behavior_clone",
    "CommanderObs", "CommanderReward", "make_commander_env",
    "InferenceTimer", "StepLogger",
]
