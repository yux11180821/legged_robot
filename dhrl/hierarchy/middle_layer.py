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

from .action import TanhNormal


class MiddleLayer(nn.Module):
    """Middle Layer (ML) -- recurrent spatiotemporal memory + command head (§4.3, Fig.2).

    A ``GRUCell`` that carries a hidden state h_t across timesteps and a linear head
    that turns the recurrent latent into a locomotion COMMAND distribution for the
    Lower Layer. The recurrence is the paper's central contribution: it lets each
    agent integrate the partial, single-neighbour observation over time to infer the
    global cooperative situation, which is why the 'No Spatiotemporal Memory' ablation
    (GRU removed) collapses to ~5-12% success (Table 1). Because UL+ML are the only
    trained components in stage 2 and the LL is frozen, the command it emits IS the
    PPO action variable, hence it is produced as a :class:`TanhNormal` in (-1, 1).

    Args:
        feature_dim: int -- width of the UL feature vector fed in (GRU input size).
        hidden_size: int -- width of the GRU hidden state h_t (default 256).
        command_dim: int -- dimensionality of the locomotion command sent to the LL
            (e.g. a target-velocity/heading command; default 3).
        use_memory: bool -- if ``False``, drop the GRU and feed the UL feature straight
            to the command head; this is the 'No Spatiotemporal Memory' ablation.
        init_log_std: float -- initial value of every component of the learnable
            log standard deviation (default -0.5 => std ~= 0.61).
    """

    def __init__(self, feature_dim: int, hidden_size: int = 256, command_dim: int = 3,
                 use_memory: bool = True, init_log_std: float = -0.5):
        super().__init__()
        self.use_memory = use_memory          # False => 'No Spatiotemporal Memory' ablation
        self.hidden_size = hidden_size
        # The GRUCell IS the spatiotemporal memory h_t; None disables it for the ablation.
        self.memory = nn.GRUCell(feature_dim, hidden_size) if use_memory else None
        # Head reads h_t when memory is on, else the raw UL feature (ablation path).
        head_in = hidden_size if use_memory else feature_dim
        self.command_mean = nn.Linear(head_in, command_dim)  # pre-tanh mean of the command
        # State-independent learnable log-std, shared across the batch (one per command dim).
        self.log_std = nn.Parameter(torch.full((command_dim,), init_log_std))

    def initial_hidden(self, batch_size: int, device) -> torch.Tensor | None:
        """Allocate the zero initial GRU hidden state h_0 for a fresh rollout.

        Called at the start of each episode/rollout so the recurrent memory begins
        from a clean state; returns ``None`` in the no-memory ablation where there is
        nothing to carry.

        Args:
            batch_size: int -- number of parallel sequences (env instances) to init.
            device: torch device on which to allocate the hidden tensor.

        Returns:
            [batch_size, hidden_size] zero tensor when memory is enabled, else ``None``.
        """
        if not self.use_memory:
            return None
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, feature: torch.Tensor, hidden: torch.Tensor | None):
        """Advance the memory one step and emit the command distribution.

        Implements the ML transition of Fig.2: ``h_{t} = GRU(feature_t, h_{t-1})``
        followed by a Gaussian command head whose samples are squashed to (-1, 1).
        The returned distribution is the PPO policy over the command (the trained
        action variable in stage 2).

        Args:
            feature: [B, feature_dim] UL perception features for this timestep.
            hidden: [B, hidden_size] previous hidden state h_{t-1}, or ``None`` to
                start from zeros / when memory is disabled.

        Returns:
            tuple ``(dist, next_hidden)`` where
              - ``dist`` is a :class:`TanhNormal` over the command, shape
                [B, command_dim] in mean/std;
              - ``next_hidden`` is the updated [B, hidden_size] state h_t, or ``None``
                in the no-memory ablation.
        """
        if self.memory is None:
            # Ablation: no recurrence, command depends only on the current feature.
            latent, next_hidden = feature, None
        else:
            if hidden is None:
                # Lazily create a zero h_{t-1} so callers may pass None on the first step.
                hidden = torch.zeros(feature.shape[0], self.hidden_size, device=feature.device)
            next_hidden = self.memory(feature, hidden)  # h_t = GRU(feature_t, h_{t-1})
            latent = next_hidden                          # command head reads the new state
        mean = self.command_mean(latent)                  # pre-tanh command mean
        std = torch.exp(self.log_std).expand_as(mean)     # broadcast the shared log-std to the batch
        return TanhNormal(mean, std), next_hidden
