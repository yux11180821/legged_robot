# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Habitat construction + simulator access helpers (lazy habitat import).

Verified attribute chain (gym_wrapper.py:164, core/env.py:383,114):
    make_gym_from_config(cfg) -> GymHabitatEnv(gym.Wrapper)
      .env  -> HabGymWrapper(gym.Wrapper)
        .env -> RLTaskEnv(RLEnv)
          ._env / .habitat_env -> Env
            ._sim / .sim -> RearrangeSim
"""

from __future__ import annotations

import numpy as np


def make_dhrl_env(
    *,
    config_name: str,
    seed: int,
    max_episode_steps: int,
    action_keys: list[str],
    enable_lateral: bool = True,
):
    """Build the gym env, restrict actions to ``action_keys``, force holonomic."""
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
        cfg.habitat.gym.action_keys = list(action_keys)
        if enable_lateral:
            for key in action_keys:
                action_cfg = cfg.habitat.task.actions.get(key, None)
                if action_cfg is not None and hasattr(action_cfg, "enable_lateral_move"):
                    action_cfg.enable_lateral_move = True
    return make_gym_from_config(cfg)


def get_sim(gym_env):
    """Reach the RearrangeSim through the wrapper chain (with fallbacks)."""
    e = gym_env
    for _ in range(6):  # unwrap gym.Wrapper layers
        if hasattr(e, "env"):
            e = e.env
        else:
            break
    inner = getattr(e, "habitat_env", None) or getattr(e, "_env", None)
    if inner is None:
        raise AttributeError(
            f"cannot find habitat Env under {type(gym_env)}; wrapper chain changed?"
        )
    sim = getattr(inner, "sim", None) or getattr(inner, "_sim", None)
    if sim is None:
        raise AttributeError(f"cannot find sim under {type(inner)}")
    return sim


def sample_navigable_point(sim, rng: np.random.Generator) -> np.ndarray:
    """Random navigable 3D point (prefers the largest indoor island)."""
    island = getattr(sim, "_largest_indoor_island_idx", None)
    pf = sim.pathfinder
    for _ in range(20):
        if island is not None:
            p = pf.get_random_navigable_point(island_index=island)
        else:
            p = pf.get_random_navigable_point()
        p = np.asarray(p, dtype=np.float32)
        if np.isfinite(p).all():
            return p
    raise RuntimeError("pathfinder failed to sample a navigable point")


def geodesic(sim, a_xz: np.ndarray, b_xz: np.ndarray, *, height: float) -> float:
    """Geodesic distance between two ground points; euclidean fallback."""
    a = np.array([a_xz[0], height, a_xz[1]], dtype=np.float32)
    b = np.array([b_xz[0], height, b_xz[1]], dtype=np.float32)
    try:
        d = float(sim.geodesic_distance(a, b))
    except Exception:
        d = float("inf")
    if not np.isfinite(d):
        d = float(np.hypot(*(np.asarray(a_xz) - np.asarray(b_xz))))
    return d


def next_waypoint(sim, start_xz: np.ndarray, goal_xz: np.ndarray, *,
                  height: float, lookahead: float = 1.5) -> np.ndarray:
    """First NavMesh shortest-path waypoint >= ``lookahead`` m from start.

    Straight-line steering toward a goal several rooms away just runs into
    walls; the manual-command gate follows the navmesh path instead.  Falls
    back to the goal itself if no path is found.
    """
    import habitat_sim

    start = np.array([start_xz[0], height, start_xz[1]], dtype=np.float32)
    goal = np.array([goal_xz[0], height, goal_xz[1]], dtype=np.float32)
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    try:
        found = sim.pathfinder.find_path(path)
    except Exception:
        found = False
    if not found or len(path.points) < 2:
        return np.asarray(goal_xz, dtype=np.float32)
    for p in path.points[1:]:
        p = np.asarray(p, dtype=np.float32)
        if np.hypot(p[0] - start[0], p[2] - start[2]) >= lookahead:
            return np.array([p[0], p[2]], dtype=np.float32)
    return np.asarray(goal_xz, dtype=np.float32)


def collision_count_from_info(info: dict) -> float:
    """Cumulative inter-agent collision count if the measure exists, else 0."""
    val = info.get("num_agents_collide", 0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0
