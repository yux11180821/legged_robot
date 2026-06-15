# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Three-layer D-HRL network structure (paper arXiv:2407.06499), Habitat edition.

Layer map (paper -> here):

  Upper Layer  (UL)  perception over exteroceptive obs e_t: relative goal +
                     NEAREST-NEIGHBOR relative position (partial observation,
                     the paper's scalability design)        -> ``UpperLayer``
  Middle Layer (ML)  recurrent planning layer: GRU hidden state h_t breaks the
                     Markov assumption (history memory)      -> inside ``DHRLActor``
  Lower Layer  (LL)  pre-trained locomotion operator (算子), trained single-agent
                     FIRST and then FROZEN; consumes proprioceptive obs p_t +
                     the ML command and outputs base velocity -> ``LowerOperator``

Ablations (paper Fig.5 / Table 1):
  dhrl          = UL + ML(RNN) + frozen LL          (full method)
  no_memory     = UL + ML(no RNN) + frozen LL       (No Spatiotemporal Memory)
  no_hierarchy  = single flat MLP -> base velocity  (No Hierarchy)

All policies emit actions through a Tanh-squashed Gaussian so outputs live in
(-1, 1)^3 -- exactly what Habitat's base-velocity step() expects (it clips each
component to [-1, 1] before scaling).  Pure torch; unit-tested in selftest.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from habitat_commander.commander_policy import TanhNormal  # noqa: E402  (tanh log-prob w/ Jacobian)

# --------------------------------------------------------------------------- #
# Dimensions (single source of truth; selftest asserts consistency).
# --------------------------------------------------------------------------- #
COMMAND_DIM = 3          # ML -> LL command: velocity mode (vx, vy, wz) or
                         # position mode (relative waypoint dx, dy + heading)
ACTION_DIM = 3           # base velocity (vx, vy, wz) in (-1, 1)
LOWER_OBS_DIM = COMMAND_DIM + ACTION_DIM  # [command(3), last_action(3)] = 6

# Per-agent exteroceptive obs e_t built by the multi-robot adapter:
#   rel_goal_ego(2) + goal_dist(1) + nn_rel_ego(2) + nn_dist(1) + last_cmd(3)
EXO_OBS_DIM = 2 + 1 + 2 + 1 + COMMAND_DIM  # = 9


def _mlp(in_dim: int, hidden: int, out_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh()]
    if out_dim is not None:
        layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Lower Layer: the operator (算子).  Stage-1 single-agent PPO, then frozen.
# --------------------------------------------------------------------------- #
class LowerOperator(nn.Module):
    """obs = [command(3), last_action(3)] -> base velocity in (-1,1)^3.

    The control mode ("position" | "velocity") changes how the *adapter* builds
    the command feature and the stage-1 reward; the network itself is identical
    (saved in the checkpoint meta).
    """

    def __init__(self, obs_dim: int = LOWER_OBS_DIM, hidden_size: int = 128,
                 action_dim: int = ACTION_DIM, init_log_std: float = -0.5):
        super().__init__()
        self.encoder = _mlp(obs_dim, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def _dist(self, obs: torch.Tensor) -> tuple[TanhNormal, torch.Tensor]:
        latent = self.encoder(obs)
        mean = self.actor_mean(latent)
        std = torch.exp(self.log_std).expand_as(mean)
        return TanhNormal(mean, std), latent

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist, latent = self._dist(obs)
        raw = dist.sample_raw()
        return raw, TanhNormal.to_env(raw), dist.log_prob_from_raw(raw), self.critic(latent).squeeze(-1)

    @torch.no_grad()
    def value_only(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encoder(obs)).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, hidden, raw: torch.Tensor):
        """(obs, hidden, raw) -> (log_prob, entropy, value); hidden is ignored
        (signature matches habitat_commander.ppo.ppo_update)."""
        dist, latent = self._dist(obs)
        return dist.log_prob_from_raw(raw), dist.entropy(), self.critic(latent).squeeze(-1)

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(obs)
        return torch.tanh(self.actor_mean(latent))


def save_lower_checkpoint(path: str, lower: LowerOperator, *, mode: str, meta: dict | None = None) -> None:
    payload = {"state_dict": lower.state_dict(), "mode": mode,
               "obs_dim": LOWER_OBS_DIM, "action_dim": ACTION_DIM}
    payload.update(meta or {})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_frozen_lower(path: str, device: torch.device) -> tuple[LowerOperator, str]:
    """Load the stage-1 operator and FREEZE it (the paper's key constraint)."""
    payload = torch.load(path, map_location=device)
    lower = LowerOperator(obs_dim=payload.get("obs_dim", LOWER_OBS_DIM))
    lower.load_state_dict(payload["state_dict"])
    lower.to(device).eval()
    for p in lower.parameters():
        p.requires_grad_(False)
    return lower, payload.get("mode", "velocity")


# --------------------------------------------------------------------------- #
# Upper + Middle layers (the per-agent actor trained in stage-2 MARL).
# --------------------------------------------------------------------------- #
class UpperLayer(nn.Module):
    """Perception layer: exteroceptive obs e_t -> feature vector."""

    def __init__(self, exo_dim: int, hidden_size: int):
        super().__init__()
        self.net = _mlp(exo_dim, hidden_size)

    def forward(self, exo: torch.Tensor) -> torch.Tensor:
        return self.net(exo)


class DHRLActor(nn.Module):
    """UL feature -> [ML GRU hidden h_t] -> command distribution (TanhNormal).

    ``use_memory=False`` is the paper's "No Spatiotemporal Memory" ablation:
    identical structure with the GRU removed.
    """

    def __init__(self, exo_dim: int = EXO_OBS_DIM, hidden_size: int = 256,
                 use_memory: bool = True, command_dim: int = COMMAND_DIM,
                 init_log_std: float = -0.5):
        super().__init__()
        self.use_memory = use_memory
        self.hidden_size = hidden_size
        self.upper = UpperLayer(exo_dim, hidden_size)
        self.memory = nn.GRUCell(hidden_size, hidden_size) if use_memory else None
        self.cmd_mean = nn.Linear(hidden_size, command_dim)
        self.log_std = nn.Parameter(torch.full((command_dim,), init_log_std))

    def initial_hidden(self, batch: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_memory:
            return None
        return torch.zeros(batch, self.hidden_size, device=device)

    def _latent(self, exo: torch.Tensor, hidden: torch.Tensor | None):
        feat = self.upper(exo)
        if self.memory is None:
            return feat, None
        if hidden is None:
            hidden = torch.zeros(exo.shape[0], self.hidden_size, device=exo.device)
        nxt = self.memory(feat, hidden)
        return nxt, nxt

    def _dist(self, latent: torch.Tensor) -> TanhNormal:
        mean = self.cmd_mean(latent)
        std = torch.exp(self.log_std).expand_as(mean)
        return TanhNormal(mean, std)

    @torch.no_grad()
    def act(self, exo: torch.Tensor, hidden: torch.Tensor | None):
        latent, nxt = self._latent(exo, hidden)
        dist = self._dist(latent)
        raw = dist.sample_raw()
        return raw, TanhNormal.to_env(raw), dist.log_prob_from_raw(raw), nxt

    def evaluate_actions(self, exo: torch.Tensor, hidden: torch.Tensor | None, raw: torch.Tensor):
        latent, _ = self._latent(exo, hidden)
        dist = self._dist(latent)
        return dist.log_prob_from_raw(raw), dist.entropy()

    @torch.no_grad()
    def deterministic_command(self, exo: torch.Tensor, hidden: torch.Tensor | None):
        latent, nxt = self._latent(exo, hidden)
        return torch.tanh(self.cmd_mean(latent)), nxt


class FlatActor(nn.Module):
    """"No Hierarchy" ablation: one flat MLP, exo obs -> base velocity directly.

    Same act/evaluate API as DHRLActor (hidden is accepted and ignored) so the
    stage-2 trainer is algorithm-agnostic.
    """

    def __init__(self, exo_dim: int = EXO_OBS_DIM, hidden_size: int = 256,
                 action_dim: int = ACTION_DIM, init_log_std: float = -0.5):
        super().__init__()
        self.use_memory = False
        self.hidden_size = hidden_size
        self.net = _mlp(exo_dim, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def initial_hidden(self, batch: int, device: torch.device):
        return None

    def _dist(self, exo: torch.Tensor) -> TanhNormal:
        mean = self.mean(self.net(exo))
        std = torch.exp(self.log_std).expand_as(mean)
        return TanhNormal(mean, std)

    @torch.no_grad()
    def act(self, exo: torch.Tensor, hidden=None):
        dist = self._dist(exo)
        raw = dist.sample_raw()
        return raw, TanhNormal.to_env(raw), dist.log_prob_from_raw(raw), None

    def evaluate_actions(self, exo: torch.Tensor, hidden, raw: torch.Tensor):
        dist = self._dist(exo)
        return dist.log_prob_from_raw(raw), dist.entropy()

    @torch.no_grad()
    def deterministic_command(self, exo: torch.Tensor, hidden=None):
        return torch.tanh(self.mean(self.net(exo))), None


class CentralCritic(nn.Module):
    """CTDE centralized critic: joint observation (all agents) -> team value.

    Used ONLY during training; execution is decentralized (each agent's actor
    sees only its own exo obs).
    """

    def __init__(self, joint_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = _mlp(joint_dim, hidden_size, out_dim=1)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.net(joint_obs).squeeze(-1)


# --------------------------------------------------------------------------- #
# Algorithm registry (paper Table 1 rows).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlgoSpec:
    label: str
    hierarchical: bool   # True -> actor outputs a command for the frozen lower
    use_memory: bool


ALGORITHMS: dict[str, AlgoSpec] = {
    "dhrl": AlgoSpec("Distributed HRL (UL+ML RNN+frozen LL)", hierarchical=True, use_memory=True),
    "no_memory": AlgoSpec("No Spatiotemporal Memory", hierarchical=True, use_memory=False),
    "no_hierarchy": AlgoSpec("No Hierarchy (flat MLP)", hierarchical=False, use_memory=False),
}


def build_actor(algorithm: str, exo_dim: int = EXO_OBS_DIM, hidden_size: int = 256):
    spec = ALGORITHMS[algorithm]
    if spec.hierarchical:
        return DHRLActor(exo_dim, hidden_size, use_memory=spec.use_memory), spec
    return FlatActor(exo_dim, hidden_size), spec


def build_lower_obs(command: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    """LL proprio obs p_t = [command, last_action].  Shapes [..., 3] each."""
    return torch.cat([command, last_action], dim=-1)
