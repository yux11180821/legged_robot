# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-agent env + simulation-backend interfaces.

Problem (§3): MDP <S, A, T, R> with N agents in M species; homogeneous species
share one policy & reward (Eq.1).  We split the two interfaces so the cooperative
TASK logic (goals, reward, observation assembly) is decoupled from the SIMULATOR
(Habitat / IsaacSim / a mock for tests):

  * ``Backend``        -- drives the simulator: apply per-agent low-level actions,
                          report agent positions / headings / proprioception.
  * ``MultiAgentEnv``  -- a task built on a backend: returns per-agent (exo, proprio),
                          a shared reward and an episode-done flag each step.

Each ``step`` returns, PER AGENT:
    exo     [N, exo_dim]      -> the trainable UL+ML actor's input  (e_t)
    proprio [N, proprio_dim]  -> the frozen LL's input              (p_t)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Simulator wrapper the task envs drive."""

    n_agents: int

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(self, actions: np.ndarray) -> None:
        """Apply per-agent low-level actions ``[N, action_dim]`` for one control step."""

    @abstractmethod
    def agent_positions(self) -> np.ndarray:
        """Ground positions ``[N, 2]``."""

    @abstractmethod
    def agent_headings(self) -> np.ndarray:
        """Headings ``[N]`` (radians)."""

    @abstractmethod
    def proprio(self) -> np.ndarray:
        """Proprioceptive obs ``[N, proprio_dim]`` (local pos/vel/torque)."""

    def collisions(self) -> int:
        """Cumulative inter-agent collision count (0 if unsupported)."""
        return 0

    def sample_navigable(self, rng: np.random.Generator) -> np.ndarray:
        """A random navigable ground point ``[2]`` (for goal sampling)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class MultiAgentEnv(ABC):
    """A cooperative task; the runner stacks E of these into an [E, N] batch."""

    n_agents: int
    exo_dim: int
    proprio_dim: int
    command_dim: int
    action_dim: int

    @abstractmethod
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """-> (exo [N, exo_dim], proprio [N, proprio_dim])."""

    @abstractmethod
    def step(self, actions: np.ndarray):
        """actions ``[N, action_dim]`` -> (exo, proprio, reward [N], done: bool, info: dict).
        ``done`` is True on success (true terminal) or time-limit (truncation, flagged in
        ``info['truncated']``); ``info`` may carry ``term_exo`` for the GAE bootstrap."""

    def close(self) -> None:
        pass
