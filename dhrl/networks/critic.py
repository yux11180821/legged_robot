# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Value functions.

  * ``DecentralizedCritic`` -- the PAPER's choice (IPPO, §4.2/§5): value estimated
    from the agent's OWN observation only.  No global state -> scalable, transfers
    to more agents (§5.4).
  * ``CentralizedCritic`` -- the CTDE BASELINE the paper compares against and beats
    (Fig.5 b/c): value from the JOINT observation of all agents.  Kept ONLY for the
    comparison; it is NOT the proposed method.
"""
from __future__ import annotations

from torch import nn


def _value_mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 1),
    )


class DecentralizedCritic(nn.Module):
    """IPPO: V(own_obs).  ``obs_dim`` = the agent's local observation dimension."""

    def __init__(self, obs_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = _value_mlp(obs_dim, hidden_size)

    def forward(self, obs):  # [B, obs_dim] -> [B]
        return self.net(obs).squeeze(-1)


class CentralizedCritic(nn.Module):
    """MAPPO baseline: V(joint_obs).  ``joint_obs_dim`` = sum of all agents' obs dims."""

    def __init__(self, joint_obs_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = _value_mlp(joint_obs_dim, hidden_size)

    def forward(self, joint_obs):  # [B, joint_obs_dim] -> [B]
        return self.net(joint_obs).squeeze(-1)
