# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Per-leg actor + centralized critic for the Go2 leg-MARL lower layer.

Reuses the bug-fixed CTDE PPO core, tanh-squashed action distribution, and
metrics harness already built for the project -- only the MuJoCo env and this
thin per-leg actor are new.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from habitat_commander.commander_policy import TanhNormal  # noqa: E402  (tanh log-prob w/ Jacobian)
from dhrl_habitat.layers import CentralCritic  # noqa: E402  (CTDE central critic; reused)

__all__ = ["LegActor", "CentralCritic", "TanhNormal"]


class LegActor(nn.Module):
    """One leg's reactive controller: proprioception -> joint action in (-1,1).

    Memoryless (the paper's lower layer is reactive; temporal memory lives in the
    middle layer).  Homogeneous: ONE LegActor is shared by all four legs and
    applied per-leg on each leg's own observation (decentralized execution).
    Matches the act/evaluate_actions interface expected by
    dhrl_habitat.marl_ppo.marl_ppo_update.
    """

    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_size: int = 128,
                 init_log_std: float = -0.5):
        super().__init__()
        self.use_memory = False
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def _dist(self, obs: torch.Tensor) -> TanhNormal:
        mean = self.mean(self.net(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return TanhNormal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden=None):
        dist = self._dist(obs)
        raw = dist.sample_raw()
        return raw, TanhNormal.to_env(raw), dist.log_prob_from_raw(raw), None

    def evaluate_actions(self, obs: torch.Tensor, hidden, raw: torch.Tensor):
        dist = self._dist(obs)
        return dist.log_prob_from_raw(raw), dist.entropy()

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor, hidden=None) -> torch.Tensor:
        return torch.tanh(self.mean(self.net(obs)))
