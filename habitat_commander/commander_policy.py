# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Upper-layer Commander policy: perception -> (vx, vy, wz) velocity command.

This is the *upper layer* of the hybrid design.  It runs in Habitat (realistic
scenes, depth/LiDAR-like sensors) and emits a holonomic base-velocity command
that drives the lower layer:

  * during Commander training, Habitat's kinematic ``base_velocity_non_cylinder``
    controller is the "ideal frozen lower layer" (it tracks the command exactly);
  * at deployment, the same command drives the IsaacLab per-leg MARL lower layer.

Key correctness choices (fixing the bugs found in the old social-nav code):

  * Actions use a **Tanh-squashed Gaussian** so the policy output lives in
    (-1, 1) -- exactly the range Habitat's base_velocity step() expects (it
    clips each component to [-1, 1] then scales by speed).  The old code declared
    a [-20, 20] action space and squashed the *mean* into [-20, 20], so ~95% of
    the range saturated.  Here the log-prob includes the tanh log-det-Jacobian,
    so the policy gradient is consistent with the executed action.

This module is PURE torch (no Habitat import) so it is unit-testable.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

_TANH_EPS = 1e-6


class TanhNormal:
    """Gaussian pushed through tanh -> support (-1, 1), with correct log_prob.

    We keep the *pre-tanh* sample ``u`` as the PPO action variable so the
    importance ratio is computed on identical variables at rollout and update.
    The environment receives ``a = tanh(u) in (-1, 1)``.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.base = Normal(mean, std)

    def sample_raw(self) -> torch.Tensor:
        return self.base.rsample()

    def log_prob_from_raw(self, u: torch.Tensor) -> torch.Tensor:
        # log p(a) = log p(u) - sum log(1 - tanh(u)^2)
        base_lp = self.base.log_prob(u).sum(dim=-1)
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + _TANH_EPS).sum(dim=-1)
        return base_lp - correction

    def entropy(self) -> torch.Tensor:
        # No closed form for the squashed entropy; use the base Normal entropy as
        # a stable proxy (standard practice for tanh-Gaussian policies).
        return self.base.entropy().sum(dim=-1)

    @staticmethod
    def to_env(u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)


class CommanderPolicy(nn.Module):
    """Perception encoder -> optional GRU memory -> (actor head, critic head).

    Operates on a flat observation vector (the Habitat adapter flattens depth +
    goal into it; an optional CNN encoder lives in ``DepthCNN`` below).  The
    optional GRU gives the Commander short-horizon memory, matching the
    "middle/planning layer" idea of the hierarchy.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 3,
        *,
        hidden_size: int = 256,
        use_memory: bool = True,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.use_memory = use_memory
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.memory = nn.GRUCell(hidden_size, hidden_size) if use_memory else None
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_memory:
            return None
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def _latent(
        self, obs: torch.Tensor, hidden: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = self.encoder(obs)
        if self.memory is None:
            return encoded, None
        if hidden is None:
            hidden = torch.zeros(obs.shape[0], self.hidden_size, device=obs.device)
        next_hidden = self.memory(encoded, hidden)
        return next_hidden, next_hidden

    def _dist(self, latent: torch.Tensor) -> TanhNormal:
        mean = self.actor_mean(latent)
        std = torch.exp(self.log_std).expand_as(mean)
        return TanhNormal(mean, std)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, hidden: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        latent, next_hidden = self._latent(obs, hidden)
        dist = self._dist(latent)
        raw = dist.sample_raw()                       # pre-tanh action (PPO variable)
        log_prob = dist.log_prob_from_raw(raw)
        value = self.critic(latent).squeeze(-1)
        env_action = TanhNormal.to_env(raw)           # in (-1, 1) -> Habitat base_velocity
        return raw, env_action, log_prob, value, next_hidden

    @torch.no_grad()
    def value_only(self, obs: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        latent, _ = self._latent(obs, hidden)
        return self.critic(latent).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, hidden: torch.Tensor | None, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, _ = self._latent(obs, hidden)
        dist = self._dist(latent)
        log_prob = dist.log_prob_from_raw(raw_actions)
        entropy = dist.entropy()
        value = self.critic(latent).squeeze(-1)
        return log_prob, entropy, value

    def deterministic_action(self, obs: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        """tanh(mean) -- used for BC distillation targets and greedy eval."""
        latent, _ = self._latent(obs, hidden)
        return torch.tanh(self.actor_mean(latent))


class DepthCNN(nn.Module):
    """Optional perception encoder for depth/LiDAR images -> feature vector.

    AdaptiveAvgPool makes it robust to the input H x W, so the Habitat adapter can
    feed raw depth without hardcoding the sensor resolution.  Use it in the
    adapter to produce part of the flat obs vector fed to ``CommanderPolicy``.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 128, pooled: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pooled, pooled)),
        )
        self.head = nn.Linear(64 * pooled * pooled, out_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: [B, C, H, W] (or [B, H, W] -> add channel)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        x = self.conv(depth)
        return torch.relu(self.head(x.flatten(1)))
