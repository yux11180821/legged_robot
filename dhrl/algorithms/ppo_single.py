# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Single-agent PPO for STAGE 1 -- pre-training the lower locomotion operator (§5.2.1).

Trains a ``LowerLayer`` in a single-agent env under one of two paradigms (set by the
LL's ``mode``):
  * position : reward = reciprocal L2 distance to a target position;
  * velocity : reward = dot(agent_velocity, target_velocity).
The paper freezes the POSITION-mode operator and reuses it for stage 2.

Same clipped-PPO + truncation-aware GAE as IPPO, but a single agent and the LL's
(proprio, command) -> action head.  ``runners/lower_trainer.py`` drives the rollout.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dhrl.memories.rollout_buffer import compute_gae_truncated  # noqa: F401  (re-exported)


def ppo_update_lower(lower, optimizer, batch: dict, *, clip_param: float = 0.2,
                     value_loss_coef: float = 0.5, entropy_coef: float = 0.0,
                     max_grad_norm: float = 0.5, ppo_epochs: int = 4,
                     num_minibatches: int = 4) -> dict[str, float]:
    """Clipped-PPO update of the ``LowerLayer``.

    ``batch`` holds flat tensors: ``proprio``, ``command``, ``raw_action``,
    ``old_log_prob``, ``advantage`` (normalized), ``ret``.
    """
    n = batch["proprio"].shape[0]
    mb = max(1, n // max(1, num_minibatches))
    stats = {"policy_loss": [], "value_loss": [], "entropy": []}

    for _ in range(ppo_epochs):
        order = torch.randperm(n, device=batch["proprio"].device)
        for start in range(0, n, mb):
            idx = order[start : start + mb]
            new_lp, entropy, value = lower.evaluate_actions(
                batch["proprio"][idx], batch["command"][idx], batch["raw_action"][idx])
            ratio = torch.exp(new_lp - batch["old_log_prob"][idx])
            adv = batch["advantage"][idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (batch["ret"][idx] - value).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(lower.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy_loss.detach().cpu()))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)
