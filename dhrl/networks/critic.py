# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Value functions.

  * ``DecentralizedCritic`` -- the PAPER's choice (IPPO, §4.2/§5): value estimated
    from the agent's OWN observation only.  No global state -> scalable, transfers
    to more agents (§5.4).
  * ``CentralizedCritic`` -- the CTDE BASELINE the paper compares against and beats
    (Fig.5 b/c): value from the JOINT observation of all agents.  Kept ONLY for the
    comparison; it is NOT the proposed method.
"""
from __future__ import annotations

from torch import nn


def _value_mlp(in_dim: int, hidden: int) -> nn.Sequential:
    """Build the shared 2-hidden-layer tanh MLP value head used by both critics.

    Factored out so the decentralized (own-obs) and centralized (joint-obs) critics share
    identical architecture and differ ONLY in their input dimension -- making the IPPO-vs-MAPPO
    comparison (Fig.5 b/c) a clean ablation of the critic's observation scope, not its capacity.

    Args:
        in_dim (int): input feature dimension (obs_dim for IPPO, joint_obs_dim for MAPPO).
        hidden (int): width of each hidden layer.

    Returns:
        nn.Sequential mapping [B, in_dim] -> [B, 1] (a scalar value per sample).
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),   # input -> hidden, tanh nonlinearity
        nn.Linear(hidden, hidden), nn.Tanh(),   # hidden -> hidden, tanh nonlinearity
        nn.Linear(hidden, 1),                   # hidden -> scalar value
    )


class DecentralizedCritic(nn.Module):
    """The paper's critic: V estimated from the agent's OWN observation only (IPPO, §4.2/§5).

    Because the value depends on no global state, the critic input dimension is independent of
    the number of agents -- the property that lets the trained policy/value transfer zero-shot
    to more agents and scale (§5.4).  This is the value head used by ``IPPO``.

    Args:
        obs_dim (int): dimension of a single agent's local (exteroceptive) observation e_t.
        hidden_size (int): width of each hidden layer in the shared value MLP.
    """

    def __init__(self, obs_dim: int, hidden_size: int = 256):
        super().__init__()
        # Value head over the agent's own obs dimension (no dependence on agent count).
        self.net = _value_mlp(obs_dim, hidden_size)

    def forward(self, obs):  # [B, obs_dim] -> [B]
        """Map a batch of single-agent observations to scalar values.

        Args:
            obs: batch of an agent's own observations, shape [B, obs_dim].

        Returns:
            V(obs) per sample, shape [B] (trailing size-1 dim squeezed off).
        """
        return self.net(obs).squeeze(-1)


class CentralizedCritic(nn.Module):
    """The CTDE BASELINE critic: V estimated from the JOINT observation of all agents.

    Implements the centralized-training side of the MAPPO baseline the paper compares against
    and beats (Fig.5 b/c): the value sees every agent's obs concatenated, so its input dim grows
    with the agent count, which is exactly why this design does NOT scale or transfer the way the
    decentralized critic does (§5.4).  Kept ONLY for the comparison; not the proposed method.

    Args:
        joint_obs_dim (int): total dimension of all agents' observations concatenated
            (i.e. sum over agents of their individual obs dims).
        hidden_size (int): width of each hidden layer in the shared value MLP.
    """

    def __init__(self, joint_obs_dim: int, hidden_size: int = 256):
        super().__init__()
        # Value head over the concatenated joint obs (input dim scales with #agents).
        self.net = _value_mlp(joint_obs_dim, hidden_size)

    def forward(self, joint_obs):  # [B, joint_obs_dim] -> [B]
        """Map a batch of joint (all-agent) observations to scalar values.

        Args:
            joint_obs: batch of concatenated all-agent observations, shape [B, joint_obs_dim].

        Returns:
            V(joint_obs) per sample, shape [B] (trailing size-1 dim squeezed off).
        """
        return self.net(joint_obs).squeeze(-1)
