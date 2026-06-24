# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Lower Layer (LL) -- pre-trained locomotion operator (paper §4.1, §5.2.1, Fig.2).

Maps (proprioceptive obs p_t, command) -> low-level action a_t.

  * STAGE 1: trained single-agent with ``ppo_single`` (it carries its own actor +
    critic heads).  Two paradigms (``mode``): 'position' (reward = reciprocal L2
    distance to a target position) and 'velocity' (reward = velocity dot product).
    The paper freezes the POSITION-mode operator.
  * STAGE 2: ``freeze()`` -> used deterministically inside the HRL policy; its
    parameters get requires_grad=False so the IPPO optimiser never touches it.
"""
from __future__ import annotations

import torch
from torch import nn

from .distributions import TanhNormal


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh())


class LowerLayer(nn.Module):
    def __init__(self, proprio_dim: int, command_dim: int, action_dim: int,
                 hidden_size: int = 256, mode: str = "position", init_log_std: float = -0.5):
        super().__init__()
        self.mode = mode
        self.encoder = _mlp(proprio_dim + command_dim, hidden_size)
        self.action_mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        self.value_head = nn.Linear(hidden_size, 1)   # used only for stage-1 PPO

    def _latent(self, proprio, command):
        return self.encoder(torch.cat([proprio, command], dim=-1))

    def _dist(self, latent) -> TanhNormal:
        mean = self.action_mean(latent)
        return TanhNormal(mean, torch.exp(self.log_std).expand_as(mean))

    # ----- stage 1: single-agent PPO -----
    @torch.no_grad()
    def act(self, proprio, command):
        latent = self._latent(proprio, command)
        dist = self._dist(latent)
        raw = dist.sample_raw()
        return raw, TanhNormal.to_env(raw), dist.log_prob_from_raw(raw), self.value_head(latent).squeeze(-1)

    def evaluate_actions(self, proprio, command, raw):
        latent = self._latent(proprio, command)
        dist = self._dist(latent)
        return dist.log_prob_from_raw(raw), dist.entropy(), self.value_head(latent).squeeze(-1)

    @torch.no_grad()
    def value_only(self, proprio, command):
        return self.value_head(self._latent(proprio, command)).squeeze(-1)

    # ----- stage 2: frozen, deterministic -----
    @torch.no_grad()
    def action(self, proprio, command):
        """Deterministic action tanh(mean) -- the frozen operator inside the HRL policy."""
        return torch.tanh(self.action_mean(self._latent(proprio, command)))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "mode": self.mode}, path)

    @classmethod
    def load_frozen(cls, path: str, *, proprio_dim: int, command_dim: int, action_dim: int,
                    hidden_size: int = 256, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        ll = cls(proprio_dim, command_dim, action_dim, hidden_size,
                 mode=payload.get("mode", "position"))
        ll.load_state_dict(payload["state_dict"])
        return ll.freeze()
