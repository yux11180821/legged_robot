# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Habitat-sim ``Backend`` (our reproduction platform).

The paper runs in IsaacSim with Ant robots; we reproduce the METHOD in Habitat-sim
(ReplicaCAD, Spot).  This wraps a multi-agent Habitat gym env behind the ``Backend``
interface so the task envs stay simulator-agnostic.  Habitat is imported lazily, so
this module imports fine on a machine without Habitat -- it only needs Habitat when a
``HabitatBackend`` is actually constructed (i.e. on the training server).

Ported from the old ``dhrl_habitat/habitat_io.py`` (verified there): the wrapper chain
``GymHabitatEnv.env.env._env._sim``, the LocalizationSensor convention
(``[x, y, z, heading]``; forward = (cos h, -sin h) in the (x, z) plane), and the
NavMesh point sampling.
"""
from __future__ import annotations

import numpy as np

from .base import Backend

_ACTION_KEY = "agent_{}_base_velocity"
_LOC_KEY = "agent_{}_localization_sensor"


class HabitatBackend(Backend):
    def __init__(self, *, config_name: str, seed: int, n_agents: int,
                 max_episode_steps: int = 500, action_dim: int = 3,
                 enable_lateral: bool = True, strip_render: bool = True):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.proprio_dim = 4  # localization [x, y, z, heading] per agent
        self._action_keys = [_ACTION_KEY.format(i) for i in range(n_agents)]
        self._loc_keys = [_LOC_KEY.format(i) for i in range(n_agents)]
        self._env = self._build(config_name, seed, max_episode_steps, enable_lateral, strip_render)
        self._sim = self._get_sim(self._env)
        self._obs: dict | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build(config_name, seed, max_episode_steps, enable_lateral, strip_render):
        import habitat
        from habitat.config import read_write
        from habitat.gym import make_gym_from_config

        cfg = habitat.get_config(config_name, overrides=[
            f"habitat.seed={seed}", f"habitat.simulator.seed={seed}",
            f"habitat.environment.max_episode_steps={max_episode_steps}",
        ])
        with read_write(cfg):
            if enable_lateral:
                for key, ac in cfg.habitat.task.actions.items():
                    if hasattr(ac, "enable_lateral_move"):
                        ac.enable_lateral_move = True
            if strip_render:  # obs is localization-only -> no per-step GPU render
                for ag in list(cfg.habitat.simulator.agents.keys()):
                    cfg.habitat.simulator.agents[ag].sim_sensors = {}
        return make_gym_from_config(cfg)

    @staticmethod
    def _get_sim(gym_env):
        e = gym_env
        for _ in range(6):
            if hasattr(e, "env"):
                e = e.env
            else:
                break
        inner = getattr(e, "habitat_env", None) or getattr(e, "_env", None)
        sim = getattr(inner, "sim", None) or getattr(inner, "_sim", None)
        if sim is None:
            raise AttributeError("could not reach RearrangeSim through the gym wrapper chain")
        return sim

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._obs = self._env.reset()

    def step(self, actions: np.ndarray) -> None:
        flat = np.asarray(actions, dtype=np.float32).reshape(-1)
        self._obs, _, _, self._info = self._env.step(flat)

    def _loc(self) -> np.ndarray:
        return np.stack([np.asarray(self._obs[k], dtype=np.float32) for k in self._loc_keys])

    def agent_positions(self) -> np.ndarray:
        loc = self._loc()
        return loc[:, [0, 2]]  # ground plane (x, z)

    def agent_headings(self) -> np.ndarray:
        return self._loc()[:, 3]

    def proprio(self) -> np.ndarray:
        return self._loc()  # [N, 4] = [x, y, z, heading]

    def collisions(self) -> int:
        try:
            return int(self._info.get("num_agents_collide", 0))
        except (AttributeError, TypeError, ValueError):
            return 0

    def sample_navigable(self, rng: np.random.Generator) -> np.ndarray:
        island = getattr(self._sim, "_largest_indoor_island_idx", None)
        pf = self._sim.pathfinder
        for _ in range(20):
            p = (pf.get_random_navigable_point(island_index=island) if island is not None
                 else pf.get_random_navigable_point())
            p = np.asarray(p, dtype=np.float32)
            if np.isfinite(p).all():
                return np.array([p[0], p[2]], dtype=np.float32)
        raise RuntimeError("pathfinder failed to sample a navigable point")

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass
