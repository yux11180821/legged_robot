from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ELU, last_activation: type[nn.Module] | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif last_activation is not None:
            layers.append(last_activation())
    return nn.Sequential(*layers)


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor
    mean: torch.Tensor


class LowerLayerCommandAdapter(nn.Module):
    """Command-level lower-layer interface.

    In IsaacSim/Gym this module should be replaced by a frozen pretrained locomotion policy
    that maps proprioceptive observation plus high-level command to joint actions. The proxy
    environments consume the command directly, so this adapter simply bounds the command.
    """

    def __init__(self, command_limit: float = 1.0, mode: Literal["velocity", "position"] = "velocity") -> None:
        super().__init__()
        self.command_limit = float(command_limit)
        self.mode = mode

    def forward(self, command: torch.Tensor) -> torch.Tensor:
        return self.command_limit * torch.tanh(command)


class DistributedHierarchicalActorCritic(nn.Module):
    """Three-layer policy from 2407.06499.

    UL encodes exteroceptive state, ML keeps recurrent spatiotemporal memory and emits a
    locomotion command, LL converts/bounds the command for the low-level controller.
    """

    def __init__(
        self,
        proprio_dim: int,
        extero_dim: int,
        command_dim: int,
        feature_dim: int,
        hidden_dim: int,
        command_limit: float = 1.0,
        lower_layer_mode: str = "velocity",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.upper = mlp([extero_dim, feature_dim, feature_dim])
        self.middle = nn.GRUCell(feature_dim, hidden_dim)
        self.command_head = mlp([hidden_dim, hidden_dim, command_dim])
        self.value_head = mlp([hidden_dim + proprio_dim, hidden_dim, 1])
        self.lower = LowerLayerCommandAdapter(command_limit, mode=lower_layer_mode)  # type: ignore[arg-type]
        self.log_std = nn.Parameter(torch.full((command_dim,), -0.5))

    def initial_hidden(self, batch_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.hidden_dim, device=device)

    def _distribution(self, proprio: torch.Tensor, extero: torch.Tensor, hidden: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        feature = self.upper(extero)
        next_hidden = self.middle(feature.reshape(-1, feature.shape[-1]), hidden.reshape(-1, hidden.shape[-1]))
        next_hidden = next_hidden.reshape(*feature.shape[:-1], self.hidden_dim)
        mean = self.lower(self.command_head(next_hidden))
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std), next_hidden, mean

    def act(self, proprio: torch.Tensor, extero: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        dist, next_hidden, mean = self._distribution(proprio, extero, hidden)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(proprio, next_hidden)
        return PolicyOutput(action, log_prob, entropy, value, next_hidden, mean)

    def evaluate_actions(self, proprio: torch.Tensor, extero: torch.Tensor, hidden: torch.Tensor, action: torch.Tensor) -> PolicyOutput:
        dist, next_hidden, mean = self._distribution(proprio, extero, hidden)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(proprio, next_hidden)
        return PolicyOutput(action, log_prob, entropy, value, next_hidden, mean)

    def value(self, proprio: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([proprio, hidden], dim=-1)).squeeze(-1)


class NoMemoryActorCritic(nn.Module):
    """Ablation for previous hierarchical baseline without recurrent memory."""

    def __init__(self, proprio_dim: int, extero_dim: int, command_dim: int, feature_dim: int, hidden_dim: int, command_limit: float = 1.0, lower_layer_mode: str = "velocity") -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = mlp([extero_dim, feature_dim, hidden_dim])
        self.command_head = mlp([hidden_dim, hidden_dim, command_dim])
        self.value_head = mlp([hidden_dim + proprio_dim, hidden_dim, 1])
        self.lower = LowerLayerCommandAdapter(command_limit, mode=lower_layer_mode)  # type: ignore[arg-type]
        self.log_std = nn.Parameter(torch.full((command_dim,), -0.5))

    def initial_hidden(self, batch_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.hidden_dim, device=device)

    def _encoded(self, extero: torch.Tensor) -> torch.Tensor:
        return self.encoder(extero)

    def act(self, proprio: torch.Tensor, extero: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        encoded = self._encoded(extero)
        mean = self.lower(self.command_head(encoded))
        dist = Normal(mean, torch.exp(self.log_std).expand_as(mean))
        action = mean if deterministic else dist.rsample()
        value = self.value(proprio, encoded)
        return PolicyOutput(action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value, encoded.detach(), mean)

    def evaluate_actions(self, proprio: torch.Tensor, extero: torch.Tensor, hidden: torch.Tensor, action: torch.Tensor) -> PolicyOutput:
        encoded = self._encoded(extero)
        mean = self.lower(self.command_head(encoded))
        dist = Normal(mean, torch.exp(self.log_std).expand_as(mean))
        value = self.value(proprio, encoded)
        return PolicyOutput(action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value, encoded, mean)

    def value(self, proprio: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        return self.value_head(torch.cat([proprio, encoded], dim=-1)).squeeze(-1)


class NoHierarchyActorCritic(nn.Module):
    """Flat single-network ablation from the paper."""

    def __init__(self, proprio_dim: int, extero_dim: int, command_dim: int, feature_dim: int, hidden_dim: int, command_limit: float = 1.0, lower_layer_mode: str = "velocity") -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = mlp([proprio_dim + extero_dim, hidden_dim, hidden_dim])
        self.command_head = nn.Linear(hidden_dim, command_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.lower = LowerLayerCommandAdapter(command_limit, mode=lower_layer_mode)  # type: ignore[arg-type]
        self.log_std = nn.Parameter(torch.full((command_dim,), -0.5))

    def initial_hidden(self, batch_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.hidden_dim, device=device)

    def _latent(self, proprio: torch.Tensor, extero: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([proprio, extero], dim=-1))

    def act(self, proprio: torch.Tensor, extero: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        latent = self._latent(proprio, extero)
        mean = self.lower(self.command_head(latent))
        dist = Normal(mean, torch.exp(self.log_std).expand_as(mean))
        action = mean if deterministic else dist.rsample()
        value = self.value_head(latent).squeeze(-1)
        return PolicyOutput(action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value, latent.detach(), mean)

    def evaluate_actions(self, proprio: torch.Tensor, extero: torch.Tensor, hidden: torch.Tensor, action: torch.Tensor) -> PolicyOutput:
        latent = self._latent(proprio, extero)
        mean = self.lower(self.command_head(latent))
        dist = Normal(mean, torch.exp(self.log_std).expand_as(mean))
        value = self.value_head(latent).squeeze(-1)
        return PolicyOutput(action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value, latent, mean)


def make_policy(variant: str, **kwargs: int | float | str) -> nn.Module:
    if variant == "dhrl":
        return DistributedHierarchicalActorCritic(**kwargs)  # type: ignore[arg-type]
    if variant == "no_memory":
        return NoMemoryActorCritic(**kwargs)  # type: ignore[arg-type]
    if variant == "no_hierarchy":
        return NoHierarchyActorCritic(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown policy variant: {variant}")
