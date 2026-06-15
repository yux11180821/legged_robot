# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Habitat adapter for the Commander: single Spot, holonomic base velocity.

Builds a single-agent Habitat gym env whose action is the holonomic
``base_velocity_non_cylinder`` command ``(vx, vy, wz)`` (the env clips each to
[-1, 1] then scales by speed -> matches the Commander's tanh output), flattens
depth + a goal-compass sensor into the policy's observation vector, and computes
a navigation reward in-trainer (progress + success - collision - slack) so we do
not depend on a fragile Habitat reward-measure config.

Habitat is imported lazily inside the functions so this module (and the pure
core) import without Habitat installed.

NOTE: the YAML config + episode dataset are the part to validate on your box
(data lives on AutoDL).  ``goal_key`` must name a (rho, phi) compass sensor that
your episodes expose -- e.g. ``pointgoal_with_gps_compass`` (Nav world) or
``target_goal_gps_compass_sensor`` (rearrange world).  See README.md.
"""

from __future__ import annotations

import numpy as np


class CommanderObs:
    """Flatten Habitat obs -> a fixed-size float vector for CommanderPolicy.

    Layout = [downsampled depth (image_size^2), goal_compass (2)].  Swap in
    ``commander_policy.DepthCNN`` for a real perception encoder (then feed a dict
    obs instead of this flat vector).
    """

    def __init__(self, depth_key: str, goal_key: str, image_size: int = 32):
        self.depth_key = depth_key
        self.goal_key = goal_key
        self.image_size = image_size
        self.feature_dim: int | None = None

    def _depth(self, arr: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=10.0, neginf=0.0)
        arr = np.squeeze(arr)
        if arr.ndim >= 2:
            yi = np.linspace(0, arr.shape[0] - 1, self.image_size).astype(np.int64)
            xi = np.linspace(0, arr.shape[1] - 1, self.image_size).astype(np.int64)
            arr = arr[np.ix_(yi, xi)]
        return arr.reshape(-1)

    def transform(self, obs: dict) -> np.ndarray:
        depth = self._depth(obs[self.depth_key])
        goal = np.asarray(obs[self.goal_key], dtype=np.float32).reshape(-1)[:2]
        out = np.concatenate([np.clip(depth, 0.0, 10.0), goal]).astype(np.float32)
        if self.feature_dim is None:
            self.feature_dim = int(out.shape[0])
        return out

    def goal_rho_phi(self, obs: dict) -> tuple[float, float]:
        g = np.asarray(obs[self.goal_key], dtype=np.float32).reshape(-1)
        return float(g[0]), float(g[1])


class CommanderReward:
    """In-trainer navigation reward (decoupled from Habitat reward measures).

    reward = progress*(prev_rho - cur_rho) + success_reward*[reached]
             - collision_penalty*[collided] - slack
    """

    def __init__(
        self,
        *,
        success_dist: float = 0.3,
        progress_weight: float = 1.0,
        success_reward: float = 10.0,
        collision_penalty: float = 0.1,
        slack: float = 0.01,
    ):
        self.success_dist = success_dist
        self.progress_weight = progress_weight
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.slack = slack
        self._prev_rho: float | None = None

    def reset(self, rho: float) -> None:
        self._prev_rho = rho

    def step(self, rho: float, collided: bool) -> tuple[float, bool]:
        prev = self._prev_rho if self._prev_rho is not None else rho
        reached = rho < self.success_dist
        reward = self.progress_weight * (prev - rho)
        reward += self.success_reward if reached else 0.0
        reward -= self.collision_penalty if collided else 0.0
        reward -= self.slack
        self._prev_rho = rho
        return float(reward), bool(reached)


def make_commander_env(
    *,
    config_name: str,
    seed: int,
    max_episode_steps: int,
    action_key: str = "base_velocity",
    enable_lateral_move: bool = True,
    project_dir: str | None = None,
):
    """Build the single-agent Spot holonomic-base-velocity gym env.

    ``enable_lateral_move=True`` turns the chosen base-velocity action into a 3-D
    holonomic command ``(vx, vy, wz)`` (matching the Commander's 3-D output).  The
    shipped social-nav config sets it to False (2-D), so we override it here, which
    lets the default ``--config-name`` reuse your already-loadable social-nav env.
    """
    import habitat
    from habitat.config import read_write
    from habitat.gym import make_gym_from_config

    overrides = [
        f"habitat.seed={seed}",
        f"habitat.simulator.seed={seed}",
        f"habitat.environment.max_episode_steps={max_episode_steps}",
    ]
    cfg = habitat.get_config(config_name, overrides=overrides)
    with read_write(cfg):
        cfg.habitat.gym.action_keys = [action_key]
        # Force holonomic (vx, vy, wz) on the controlled action if available.
        action_cfg = cfg.habitat.task.actions.get(action_key, None)
        if action_cfg is not None and hasattr(action_cfg, "enable_lateral_move"):
            action_cfg.enable_lateral_move = enable_lateral_move
    return make_gym_from_config(cfg)


def collision_from_info(info: dict) -> bool:
    """Best-effort collision flag from Habitat info (measure names vary)."""
    for key in ("did_collide", "num_agents_collide", "collisions"):
        if key in info:
            val = info[key]
            if isinstance(val, dict):
                val = val.get("is_collision", val.get("count", 0))
            try:
                return bool(float(val) > 0)
            except (TypeError, ValueError):
                continue
    return False
