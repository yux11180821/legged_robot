# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Task -- Corridor Crossing (paper Fig.1b, homogeneous).

Agents must pass through a narrow corridor; only one can pass at a time, so
"yield / concession" behaviour has to emerge.  We realise it with the swap layout:
each agent's goal is another agent's start, forcing them to cross each other.

Reward is SHARED across agents (Eq.1, homogeneous):
    sum_i progress_i (toward own goal)  + first-arrival bonus per agent
    + all-arrived bonus                 - inter-agent collision penalty  - slack
Episode ends on all-arrived (true terminal) or time limit (truncation -> the runner
bootstraps V(term_exo); ``info['truncated']`` / ``info['term_exo']``).

The task is decoupled from the simulator via a ``Backend`` (so it is testable with a
mock); positions/headings/proprio/collisions come from the backend.
"""
from __future__ import annotations

import numpy as np

from . import obs as O
from .base import Backend, MultiAgentEnv


class CorridorCrossing(MultiAgentEnv):
    def __init__(self, backend: Backend, *, proprio_dim: int, command_dim: int,
                 action_dim: int, max_steps: int = 500, success_dist: float = 0.5,
                 dist_norm: float = 10.0, progress_w: float = 1.0, arrival_bonus: float = 2.5,
                 all_arrived_bonus: float = 10.0, collision_pen: float = 0.25,
                 slack: float = 0.01, rng: np.random.Generator | None = None):
        self.backend = backend
        self.n_agents = backend.n_agents
        self.proprio_dim = proprio_dim
        self.command_dim = command_dim
        self.action_dim = action_dim
        self.exo_dim = 6  # goal-ego(3) + nearest-neighbour-ego(3)
        self.max_steps = max_steps
        self.success_dist = success_dist
        self.dist_norm = dist_norm
        self.progress_w = progress_w
        self.arrival_bonus = arrival_bonus
        self.all_arrived_bonus = all_arrived_bonus
        self.collision_pen = collision_pen
        self.slack = slack
        self.rng = rng or np.random.default_rng()
        self._goals = None

    def _exo(self) -> np.ndarray:
        return O.build_exteroceptive(self.backend.agent_positions(), self.backend.agent_headings(),
                                     goals=self._goals, dist_norm=self.dist_norm)

    def _dists(self) -> np.ndarray:
        return np.linalg.norm(self.backend.agent_positions() - self._goals, axis=-1)

    def _sample_goals(self) -> None:
        # swap: goal_i = start of agent (i+1) % N -> forced crossing
        self._goals = np.roll(self.backend.agent_positions().copy(), shift=-1, axis=0)

    def reset(self):
        self.backend.reset()
        self._sample_goals()
        self._prev_dist = self._dists()
        self._arrived = np.zeros(self.n_agents, dtype=bool)
        self._prev_coll = 0
        self._t = 0
        return self._exo(), self.backend.proprio()

    def step(self, actions):
        self.backend.step(np.asarray(actions, dtype=np.float32))
        self._t += 1

        cur = self._dists()
        progress = float(np.sum(self._prev_dist - cur))      # team progress (shared)
        self._prev_dist = cur

        newly = (cur < self.success_dist) & (~self._arrived)
        self._arrived |= cur < self.success_dist
        all_arrived = bool(self._arrived.all())

        coll = self.backend.collisions()
        coll_delta = max(0, coll - self._prev_coll)
        self._prev_coll = coll

        shared = (self.progress_w * progress
                  + self.arrival_bonus * float(np.sum(newly))
                  + (self.all_arrived_bonus if all_arrived else 0.0)
                  - self.collision_pen * coll_delta
                  - self.slack)
        reward = np.full(self.n_agents, shared, dtype=np.float32)  # homogeneous shared reward

        truncated = (self._t >= self.max_steps) and not all_arrived
        done = all_arrived or self._t >= self.max_steps
        exo = self._exo()
        info = {"truncated": truncated, "success": all_arrived,
                "arrived": int(self._arrived.sum()), "term_exo": exo}
        return exo, self.backend.proprio(), reward, done, info

    def close(self):
        self.backend.close()
