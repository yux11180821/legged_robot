# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""PPO pieces for the Commander, with the two bug fixes from the old code.

  1. TRUNCATION-AWARE GAE: episodes that end because of the time limit are
     bootstrapped (V(terminal_obs) is added), while genuine task terminals are
     not.  The old code treated every ``done`` as a true terminal, dropping the
     bootstrap at every time-limit truncation (the dominant termination cause).

  2. Action squashing into the env's real range is handled in CommanderPolicy
     (TanhNormal); here we just keep the importance ratio consistent.

``compute_gae_truncated`` is pure numpy and unit-tested (see selftest.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def compute_gae_truncated(
    rewards: np.ndarray,        # [T, N]
    values: np.ndarray,         # [T, N]
    dones: np.ndarray,          # [T, N] 1.0 if the episode ended at this step (then env was reset)
    bootstrap_values: np.ndarray,  # [T, N] V(terminal_obs) on TRUNCATION, else 0.0
    last_value: np.ndarray,     # [N] V of the obs after the final rollout step
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncation-aware GAE.

    For a step that ends an episode (``dones[t] == 1``):
      * the delta bootstraps from ``bootstrap_values[t]`` -- which the caller sets
        to ``V(terminal_obs)`` for a TIME-LIMIT truncation and to ``0`` for a
        genuine terminal;
      * the GAE recursion is cut by ``(1 - dones[t])`` so advantages never leak
        across the episode boundary.
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        # On an episode boundary, the correct "next value" is the bootstrap value
        # (terminal_value on truncation, 0 on true terminal), NOT V(reset_obs).
        next_value = np.where(dones[t] > 0.5, bootstrap_values[t], next_value)
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


@dataclass
class PPOBatch:
    obs: torch.Tensor
    hidden: torch.Tensor | None
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    *,
    clip_param: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    num_minibatches: int = 4,
) -> dict[str, float]:
    batch_size = batch.obs.shape[0]
    minibatch_size = max(1, batch_size // max(1, num_minibatches))
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(ppo_epochs):
        order = torch.randperm(batch_size, device=batch.obs.device)
        for start in range(0, batch_size, minibatch_size):
            idx = order[start : start + minibatch_size]
            mb_hidden = None if batch.hidden is None else batch.hidden[idx]
            new_log_probs, entropy, values = policy.evaluate_actions(
                batch.obs[idx], mb_hidden, batch.raw_actions[idx]
            )
            ratio = torch.exp(new_log_probs - batch.old_log_probs[idx])
            adv = batch.advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (batch.returns[idx] - values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch.old_log_probs[idx] - new_log_probs).mean()
            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy_loss.detach().cpu()))
            stats["approx_kl"].append(float(approx_kl.detach().cpu()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
