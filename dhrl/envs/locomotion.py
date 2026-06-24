# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Single-agent locomotion-operator pre-training task (paper §5.2.1).

The lower layer is trained to TRACK a sampled command under one of two paradigms:
  * position : reward = reciprocal L2 distance to a target position;
  * velocity : reward = dot(agent_velocity, target_velocity).

NOTE: the dynamics here are a small synthetic double-integrator -- a runnable,
paper-faithful-reward placeholder for the real sim-based locomotion task (IsaacSim
Ant / Habitat Spot), which swaps in via a backend later.  It exists so stage 1's
training loop (``runners/lower_trainer.py``) runs end-to-end without a simulator.
"""
from __future__ import annotations

import numpy as np


class LocomotionEnv:
    def __init__(self, *, mode: str = "position", dim: int = 2, max_steps: int = 200,
                 dt: float = 0.1, rng: np.random.Generator | None = None):
        assert mode in ("position", "velocity")
        self.mode = mode
        self.action_dim = dim
        self.command_dim = dim
        self.proprio_dim = 2 * dim            # [position, velocity]
        self.max_steps = max_steps
        self.dt = dt
        self.rng = rng or np.random.default_rng()

    def reset(self):
        self.pos = np.zeros(self.action_dim, np.float32)
        self.vel = np.zeros(self.action_dim, np.float32)
        self.target = self.rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
        self.t = 0
        return self._proprio(), self.target.copy()

    def _proprio(self):
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.vel = self.vel + a * self.dt          # action ~ acceleration
        self.pos = self.pos + self.vel * self.dt
        self.t += 1
        if self.mode == "position":
            reward = 1.0 / (1.0 + float(np.linalg.norm(self.pos - self.target)))
        else:
            reward = float(np.dot(self.vel, self.target))
        done = self.t >= self.max_steps
        proprio, command = self._proprio(), self.target.copy()
        info = {"truncated": done, "term_proprio": proprio, "term_command": command}
        return proprio, command, reward, done, info

    def close(self):
        pass
