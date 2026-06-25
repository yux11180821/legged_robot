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
    """Single-agent command-tracking env that pre-trains the Lower-Layer operator (§5.2.1).

    Why: stage 1 of the two-stage recipe trains the locomotion operator (the Lower Layer,
    §4.3/Fig.2) to TRACK a sampled command, then FREEZES it before stage-2 cooperation
    training.  The operator maps proprioception p_t + command -> action a_t, and is rewarded
    under one of the paper's two command paradigms: ``position`` (reward = reciprocal L2
    distance to a target position) or ``velocity`` (reward = dot(velocity, target velocity)).
    The dynamics here are a deliberately tiny synthetic double-integrator standing in for
    the real sim-based legged robot (IsaacSim Ant / Habitat Spot); the rewards are kept
    paper-faithful so the stage-1 loop runs end-to-end without a simulator and the operator
    can later be swapped for a real one via a backend.
    """

    def __init__(self, *, mode: str = "position", dim: int = 2, max_steps: int = 200,
                 dt: float = 0.1, rng: np.random.Generator | None = None):
        """Configure the single-agent pre-training env (paradigm, dimensionality, horizon).

        Why: selects which of the two paper command paradigms (§5.2.1) defines the reward
        and sizes the proprio/command/action vectors so the Lower-Layer policy network can
        be built to match.  ``proprio_dim`` is 2*dim because the double-integrator state is
        the concatenation [position, velocity].

        Args:
            mode: str -- "position" or "velocity"; selects the command paradigm / reward.
            dim: int -- dimensionality of the workspace; sets action_dim and command_dim.
            max_steps: int -- episode horizon; hitting it ends the episode (a truncation).
            dt: float -- integration timestep for the double-integrator dynamics.
            rng: optional NumPy Generator for reproducible target sampling (default: fresh).

        Returns:
            None.
        """
        assert mode in ("position", "velocity")  # only the two paper paradigms are valid
        self.mode = mode
        self.action_dim = dim
        self.command_dim = dim
        self.proprio_dim = 2 * dim            # [position, velocity]
        self.max_steps = max_steps
        self.dt = dt
        self.rng = rng or np.random.default_rng()

    def reset(self):
        """Start a new episode: zero the body state and draw a fresh target command.

        Why: each episode re-samples the command the Lower Layer must track (a random
        target in [-1, 1]^dim) and returns the first (proprio, command) pair, mirroring how
        stage-1 training exposes the operator to many sampled commands.

        Returns:
            tuple (proprio, command):
                proprio [2*dim] float -- initial proprioceptive state p_t = [pos, vel] = 0.
                command [dim]   float -- the per-episode target the operator must track.
        """
        self.pos = np.zeros(self.action_dim, np.float32)     # start at the origin, at rest
        self.vel = np.zeros(self.action_dim, np.float32)
        self.target = self.rng.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)  # sampled command
        self.t = 0
        return self._proprio(), self.target.copy()

    def _proprio(self):
        """Pack the current body state into the proprioceptive observation vector p_t.

        Why: p_t = [position, velocity] is exactly the Lower-Layer operator's input
        (proprioception, §4.3); kept as one helper so reset/step build it identically.

        Returns:
            [2*dim] float -- concatenated [position, velocity] proprioceptive vector.
        """
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, action):
        """Integrate the dynamics one step and return the paradigm-specific reward.

        Why: the action is treated as acceleration driving a double-integrator (vel += a*dt,
        pos += vel*dt), and the reward implements the chosen paper paradigm (§5.2.1) --
        position: 1/(1+||pos-target||) (reciprocal L2 distance, bounded in (0,1]); velocity:
        dot(vel, target) (move fast along the commanded direction).  The episode ends purely
        on the time limit, so ``done`` is always a truncation, and ``info`` carries the
        terminal proprio/command for a truncation-aware value bootstrap.

        Args:
            action: [dim] float -- the operator's action a_t (interpreted as acceleration,
                clipped to [-1, 1]).

        Returns:
            tuple (proprio, command, reward, done, info):
                proprio [2*dim] float -- next proprioceptive state p_t = [pos, vel].
                command [dim]   float -- the (unchanged) per-episode target command.
                reward  float         -- paradigm-specific tracking reward for this step.
                done    bool          -- True at the time limit (always a TRUNCATION here).
                info    dict          -- ``truncated`` plus ``term_proprio`` / ``term_command``
                                         (terminal obs for the GAE value bootstrap).
        """
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)  # actuation limits
        self.vel = self.vel + a * self.dt          # action ~ acceleration
        self.pos = self.pos + self.vel * self.dt   # semi-implicit Euler position update
        self.t += 1
        if self.mode == "position":
            reward = 1.0 / (1.0 + float(np.linalg.norm(self.pos - self.target)))  # reciprocal L2 distance (position paradigm)
        else:
            reward = float(np.dot(self.vel, self.target))  # velocity-along-target (velocity paradigm)
        done = self.t >= self.max_steps                # time-limit only -> truncation
        proprio, command = self._proprio(), self.target.copy()
        info = {"truncated": done, "term_proprio": proprio, "term_command": command}  # terminal obs for bootstrap
        return proprio, command, reward, done, info

    def close(self):
        """No-op cleanup hook (this synthetic env holds no external resources).

        Why: mirrors the env interface so callers can close it uniformly; there is nothing
        to release for the in-process double-integrator.

        Returns:
            None.
        """
        pass
