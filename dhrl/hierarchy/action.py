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
    """Squashed-Gaussian distribution: a diagonal Normal pushed through ``tanh``.

    Wraps a ``torch.distributions.Normal`` whose samples are squashed by ``tanh`` so
    that every action component is confined to the open interval (-1, 1). Both the
    Middle Layer's locomotion COMMAND (the PPO action variable trained by IPPO in
    stage 2) and the frozen Lower Layer's low-level ACTION are emitted as instances
    of this class, which is why it lives in a paper-agnostic, import-free module:
    keeping the PPO importance ratio consistent requires that the SAME pre-tanh
    variable ``u`` be sampled at rollout time and re-scored at update time, and the
    tanh squashing guarantees the executed action matches the env's bounded action
    space (each component is clipped to [-1, 1] then rescaled by the env).

    Args:
        mean: [..., action_dim] per-component means of the underlying Gaussian (the
            network's pre-tanh ``mean`` head output).
        std: [..., action_dim] per-component standard deviations (``exp(log_std)``);
            must be positive and broadcastable to ``mean``.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # ``base`` is the un-squashed diagonal Gaussian; the tanh squashing is applied
        # lazily by the helpers below so we can keep ``u`` (pre-tanh) for PPO scoring.
        self.base = Normal(mean, std)

    def sample_raw(self) -> torch.Tensor:
        """Draw a reparameterized PRE-tanh sample ``u`` (the PPO action variable).

        Returns the un-squashed Gaussian sample rather than the squashed action so
        that the PPO importance ratio can be evaluated on identical variables at
        rollout and update; ``rsample`` keeps the path differentiable (reparameter-
        ization trick), which matters for the stage-1 single-agent PPO that backprops
        through the action head.

        Returns:
            [..., action_dim] pre-tanh sample ``u`` (unbounded real values).
        """
        return self.base.rsample()  # reparameterized so grads can flow through u

    def log_prob_from_raw(self, u: torch.Tensor) -> torch.Tensor:
        """Log-density of the SQUASHED action that corresponds to pre-tanh ``u``.

        Applies the change-of-variables correction for the ``tanh`` squashing so the
        returned value is ``log p(a)`` with ``a = tanh(u)``, not ``log p(u)``. This is
        what makes the policy gradient consistent with the action actually executed in
        the env; both stage-1 LL PPO and stage-2 UL+ML IPPO call this to score stored
        raw samples during the clipped-objective update.

        Args:
            u: [..., action_dim] pre-tanh sample previously returned by
                :meth:`sample_raw` (the variable the ratio is computed on).

        Returns:
            [...] summed-over-components log-probability of ``a = tanh(u)`` (the
            ``action_dim`` axis is reduced away, matching a single scalar per sample).
        """
        # log p(a) = log p(u) - sum log(1 - tanh(u)^2)  (tanh log-det-Jacobian term)
        base_lp = self.base.log_prob(u).sum(dim=-1)  # per-component Gaussian log-probs, summed
        # _TANH_EPS guards the log against exactly-saturated tanh (|tanh|->1 => arg->0).
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + _TANH_EPS).sum(dim=-1)
        return base_lp - correction  # subtract the Jacobian to get the squashed density

    def entropy(self) -> torch.Tensor:
        """Entropy proxy used for the PPO entropy-bonus regularizer.

        The squashed (tanh-Gaussian) distribution has no closed-form differential
        entropy, so we return the entropy of the underlying Normal -- the standard,
        numerically stable surrogate used to keep the policy exploratory in both
        training stages.

        Returns:
            [...] summed-over-components base-Normal entropy (one scalar per sample).
        """
        # No closed form for the squashed entropy; the base Normal entropy is the
        # standard stable proxy for tanh-Gaussian policies.
        return self.base.entropy().sum(dim=-1)

    @staticmethod
    def to_env(u: torch.Tensor) -> torch.Tensor:
        """Map a pre-tanh sample ``u`` to the bounded env action ``a = tanh(u)``.

        The squashing step that turns the PPO action variable into something the env
        can consume; kept separate from sampling so callers can store ``u`` for the
        importance ratio while sending ``tanh(u)`` to the simulator.

        Args:
            u: [..., action_dim] pre-tanh sample.

        Returns:
            [..., action_dim] action with every component in (-1, 1).
        """
        return torch.tanh(u)

    @staticmethod
    def deterministic(mean: torch.Tensor) -> torch.Tensor:
        """Greedy (mode-like) action ``tanh(mean)`` -- used at EVAL, no sampling.

        At evaluation the network is locked and we want the most-likely action rather
        than a stochastic draw; squashing the Gaussian mean gives the deterministic
        command/action used by the deterministic rollout paths of the LL and HRL
        policy.

        Args:
            mean: [..., action_dim] pre-tanh Gaussian mean from the network head.

        Returns:
            [..., action_dim] deterministic action ``tanh(mean)`` in (-1, 1).
        """
        return torch.tanh(mean)
