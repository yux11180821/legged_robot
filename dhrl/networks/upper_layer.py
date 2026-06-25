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
    """Upper Layer (UL) perception MLP -- the top of the 3-layer HRL stack (§4.1, Fig.2).

    A small two-layer ``tanh`` MLP that encodes the exteroceptive observation e_t into
    a dense feature vector consumed by the Middle Layer. e_t is deliberately PARTIAL
    (Fig.3, §4.2): it contains the environmental/terrain perception plus only the
    relative position of the SINGLE nearest neighbour, never the global state of all
    peers. That partial-observation design is precisely what keeps the controller
    decentralized and scalable to more agents without re-training.

    Args:
        exo_dim: int -- dimensionality of the exteroceptive observation e_t (terrain
            perception features concatenated with the nearest-neighbour relative
            position); sets the input width of the first linear layer.
        hidden_size: int -- width of the hidden ``tanh`` layer (default 256).
        feature_dim: int | None -- width of the emitted feature vector; defaults to
            ``hidden_size`` when ``None`` so the UL and ML hidden widths match.
    """

    def __init__(self, exo_dim: int, hidden_size: int = 256, feature_dim: int | None = None):
        super().__init__()
        # Fall back to hidden_size so UL output and ML GRU input widths line up by default.
        self.feature_dim = feature_dim or hidden_size
        self.net = nn.Sequential(
            nn.Linear(exo_dim, hidden_size), nn.Tanh(),          # encode e_t
            nn.Linear(hidden_size, self.feature_dim), nn.Tanh(),  # project to the ML feature space
        )

    def forward(self, exo):  # exo: [B, exo_dim] -> [B, feature_dim]
        """Encode a batch of exteroceptive observations into UL features.

        The perception step of Fig.2: turns e_t into the spatial feature that the
        Middle Layer's recurrent memory integrates over time.

        Args:
            exo: [B, exo_dim] batch of exteroceptive observations e_t (terrain +
                nearest-neighbour relative position).

        Returns:
            [B, feature_dim] dense perception feature forwarded to the Middle Layer.
        """
        return self.net(exo)
