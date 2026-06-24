# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""TanhNormal squashed-Gaussian action distribution.

A Gaussian pushed through ``tanh`` so samples live in (-1, 1) -- exactly the range
the env's continuous action expects (it clips each component to [-1, 1] then scales).
The PPO action variable is the PRE-tanh sample ``u`` so the importance ratio is
computed on identical variables at rollout and update; the env receives
``a = tanh(u)``.  ``log_prob`` includes the tanh log-det-Jacobian, so the policy
gradient is consistent with the executed action.

Pure torch; no project imports, so it is trivially unit-testable.
"""
from __future__ import annotations

import torch
from torch.distributions import Normal

_TANH_EPS = 1e-6


class TanhNormal:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.base = Normal(mean, std)

    def sample_raw(self) -> torch.Tensor:
        """Pre-tanh sample ``u`` (the PPO action variable), reparameterized."""
        return self.base.rsample()

    def log_prob_from_raw(self, u: torch.Tensor) -> torch.Tensor:
        # log p(a) = log p(u) - sum log(1 - tanh(u)^2)
        base_lp = self.base.log_prob(u).sum(dim=-1)
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + _TANH_EPS).sum(dim=-1)
        return base_lp - correction

    def entropy(self) -> torch.Tensor:
        # No closed form for the squashed entropy; the base Normal entropy is the
        # standard stable proxy for tanh-Gaussian policies.
        return self.base.entropy().sum(dim=-1)

    @staticmethod
    def to_env(u: torch.Tensor) -> torch.Tensor:
        """Map a pre-tanh sample to the env action in (-1, 1)."""
        return torch.tanh(u)

    @staticmethod
    def deterministic(mean: torch.Tensor) -> torch.Tensor:
        """Greedy action ``tanh(mean)`` -- used at EVAL (network locked, no sampling)."""
        return torch.tanh(mean)
