# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Upper Layer (UL) -- perception (paper §4.1, Fig.2).

Input : exteroceptive obs e_t = [ environmental perception (height/terrain array),
        NEAREST-NEIGHBOUR relative position ]  (Fig.3 -- PARTIAL observation; each
        agent sees only its closest peer, which is what lets the method scale).
Output: a feature vector forwarded to the Middle Layer.
"""
from __future__ import annotations

from torch import nn


class UpperLayer(nn.Module):
    def __init__(self, exo_dim: int, hidden_size: int = 256, feature_dim: int | None = None):
        super().__init__()
        self.feature_dim = feature_dim or hidden_size
        self.net = nn.Sequential(
            nn.Linear(exo_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, self.feature_dim), nn.Tanh(),
        )

    def forward(self, exo):  # exo: [B, exo_dim] -> [B, feature_dim]
        return self.net(exo)
