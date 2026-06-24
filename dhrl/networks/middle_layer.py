# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Middle Layer (ML) -- spatiotemporal memory (paper §4.3, Fig.2).

A RECURRENT layer maintaining hidden state h_t: input = UL feature, output = a
locomotion COMMAND distribution for the LL plus the next hidden state.  The RNN is
the paper's core contribution -- the 'No Spatiotemporal Memory' ablation (GRU
removed) collapses to ~5-12% success (Table 1).

The command is the PPO action variable (UL+ML are trained, the LL is frozen), so
it is emitted as a ``TanhNormal`` distribution in (-1, 1).
"""
from __future__ import annotations

import torch
from torch import nn

from .distributions import TanhNormal


class MiddleLayer(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int = 256, command_dim: int = 3,
                 use_memory: bool = True, init_log_std: float = -0.5):
        super().__init__()
        self.use_memory = use_memory          # False => 'No Spatiotemporal Memory' ablation
        self.hidden_size = hidden_size
        self.memory = nn.GRUCell(feature_dim, hidden_size) if use_memory else None
        head_in = hidden_size if use_memory else feature_dim
        self.command_mean = nn.Linear(head_in, command_dim)
        self.log_std = nn.Parameter(torch.full((command_dim,), init_log_std))

    def initial_hidden(self, batch_size: int, device) -> torch.Tensor | None:
        if not self.use_memory:
            return None
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, feature: torch.Tensor, hidden: torch.Tensor | None):
        """feature: [B, feature_dim] -> (TanhNormal over command, next_hidden|None)."""
        if self.memory is None:
            latent, next_hidden = feature, None
        else:
            if hidden is None:
                hidden = torch.zeros(feature.shape[0], self.hidden_size, device=feature.device)
            next_hidden = self.memory(feature, hidden)
            latent = next_hidden
        mean = self.command_mean(latent)
        std = torch.exp(self.log_std).expand_as(mean)
        return TanhNormal(mean, std), next_hidden
