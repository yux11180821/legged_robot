# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Command sources for the layered controller: MANUAL commands first.

MEETING DECISION (元宝纪要): the VLM/large-model command route is DEPRIORITIZED
-- the advisor rejected it because the model's per-step execution error (~0.5)
accumulates over a sequence and destabilizes training, and inference is slow.
The plan is to validate HAND-CRAFTED ("人工指令") commands against the frozen
lower operator FIRST; "if the manual command does not work, the VLM command
cannot work either."

So this module's primary export is ``ManualCommandPolicy`` -- a dependency-free,
deterministic command generator (points toward the goal from a (rho, phi)
compass).  Use it to (a) drive the frozen lower operator for validation, and
(b) optionally behavior-clone a warm start before learning the upper layer.

The ``VLMTeacher`` Protocol is kept as a future hook ONLY: if/when the VLM route
is revisited, implement it (render frame -> prompt VLM -> parse (vx, vy, wz)) and
swap it in wherever ``ManualCommandPolicy`` is used.  ``OracleGoalTeacher`` is an
alias of ``ManualCommandPolicy`` for back-compat.

Pure torch/numpy (no Habitat import).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import nn


class VLMTeacher(Protocol):
    """FUTURE hook only (VLM route deprioritized, see module docstring).

    Maps observations -> target velocity command in [-1, 1]^3 = (vx, vy, wz)."""

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:  # [B, 3]
        ...


class ManualCommandPolicy:
    """Hand-crafted ("人工指令") command: steer toward the goal via the compass.

    This is the PRIMARY command source the advisor asked to validate first.
    Expects ``obs["pointgoal_with_gps_compass"] = [B, 2] = (rho, phi)`` where rho
    is distance-to-goal and phi is the bearing (radians, 0 = straight ahead, +ccw)
    -- the standard Habitat PointGoal-with-GPS-Compass convention.

    It is deterministic and instant (no model), so it is also the natural
    inference-time baseline to contrast against a slow VLM controller.
    """

    def __init__(self, max_lin: float = 1.0, max_ang: float = 1.0, turn_gain: float = 1.5):
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.turn_gain = turn_gain

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        pg = np.asarray(obs["pointgoal_with_gps_compass"], dtype=np.float32)
        if pg.ndim == 1:
            pg = pg[None, :]
        rho, phi = pg[:, 0], pg[:, 1]
        # Holonomic command toward the goal, in the base frame.
        vx = np.cos(phi) * np.clip(rho, 0.0, 1.0)
        vy = np.sin(phi) * np.clip(rho, 0.0, 1.0)
        wz = np.clip(self.turn_gain * phi, -1.0, 1.0)
        cmd = np.stack([vx, vy, wz], axis=-1) * np.array(
            [self.max_lin, self.max_lin, self.max_ang], dtype=np.float32
        )
        return np.clip(cmd, -1.0, 1.0).astype(np.float32)


# Back-compat alias (the old name).  Prefer ManualCommandPolicy.
OracleGoalTeacher = ManualCommandPolicy


def behavior_clone(
    policy: nn.Module,
    obs: torch.Tensor,                 # [M, obs_dim] flattened observations
    target_actions: torch.Tensor,      # [M, 3] in [-1, 1]
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Behavior-clone the teacher into ``policy``'s mean (warm start before PPO).

    Loss = MSE( tanh(policy_mean(obs)), target ).  Uses the policy's perception
    trunk + actor head, so PPO continues from an aligned initialization.  Memory
    is bypassed here (single-step BC); PPO learns the recurrent dynamics.
    """
    device = device or next(policy.parameters()).device
    obs = obs.to(device)
    target_actions = target_actions.to(device).clamp(-0.999, 0.999)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = obs.shape[0]
    last = {}
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            pred = policy.deterministic_action(obs[idx], None)  # tanh(mean)
            loss = (pred - target_actions[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last = {"bc_epoch": epoch, "bc_mse": float(np.mean(losses))}
    return last
