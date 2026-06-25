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

from memories.rollout_buffer import compute_gae_truncated  # noqa: F401  (re-exported)


def ppo_update_lower(lower, optimizer, batch: dict, *, clip_param: float = 0.2,
                     value_loss_coef: float = 0.5, entropy_coef: float = 0.0,
                     max_grad_norm: float = 0.5, ppo_epochs: int = 4,
                     num_minibatches: int = 4) -> dict[str, float]:
    """Run one clipped-PPO optimization phase over a stage-1 rollout of the ``LowerLayer``.

    This is the single-agent learner that pre-trains the Lower Layer (LL) locomotion
    operator in stage 1 (§5.2.1): the LL is the proprio+command -> action head that the
    paper later FREEZES and reuses unchanged inside the hierarchy for stage 2. Because
    stage 1 is single-agent there is no IPPO multi-agent bookkeeping here -- it is the
    plain clipped-PPO surrogate of Schulman et al. (2017) applied to the LL's own actor
    and its scalar value head. ``entropy_coef`` defaults to 0 because the locomotion
    operator is trained toward a deterministic gait rather than a high-entropy explorer.

    Args:
        lower: the ``LowerLayer`` module being trained. Exposes ``evaluate_actions`` (recomputes
            log-prob / entropy / value under the current params) and ``parameters`` for grad clipping.
        optimizer: the Adam optimizer over ``lower.parameters()``.
        batch (dict): flat (already time-x-env-collapsed) rollout tensors, each first dim ``n``:
            ``proprio``     [n, proprio_dim] proprioceptive state p_t fed to the LL;
            ``command``     [n, command_dim] the locomotion command driving the LL (in stage 1 this
                            comes from the env target, in stage 2 it would come from the ML);
            ``raw_action``  [n, action_dim]  the pre-squash action sample whose log-prob is re-scored;
            ``old_log_prob``[n]              log pi_old(a|s) captured at collection time (PPO ratio denom);
            ``advantage``   [n]              GAE advantage, already normalized by ``normalize``;
            ``ret``         [n]              bootstrapped return target for the value head.
        clip_param (float): PPO ratio clip epsilon (the 1 +/- eps trust region).
        value_loss_coef (float): weight on the critic MSE term in the joint loss.
        entropy_coef (float): weight on the entropy bonus (0 -> no exploration bonus, see above).
        max_grad_norm (float): global L2 grad-norm clip for stability.
        ppo_epochs (int): number of full passes over the batch per update (data reuse).
        num_minibatches (int): how many shuffled minibatches to split each epoch into.

    Returns:
        dict[str, float]: mean ``policy_loss`` (surrogate), ``value_loss`` (MSE), and ``entropy``
        across all minibatch steps -- scalars logged to the stage-1 metrics CSV.
    """
    n = batch["proprio"].shape[0]                          # flat batch size (T * single env)
    mb = max(1, n // max(1, num_minibatches))              # minibatch size; guard against n < num_minibatches
    stats = {"policy_loss": [], "value_loss": [], "entropy": []}

    for _ in range(ppo_epochs):                            # multiple passes => PPO sample reuse
        order = torch.randperm(n, device=batch["proprio"].device)  # reshuffle each epoch to decorrelate minibatches
        for start in range(0, n, mb):
            idx = order[start : start + mb]                # indices for this minibatch
            # Re-score the stored actions under the CURRENT policy to get the new log-prob,
            # the entropy of the current action distribution, and the current value estimate.
            new_lp, entropy, value = lower.evaluate_actions(
                batch["proprio"][idx], batch["command"][idx], batch["raw_action"][idx])
            ratio = torch.exp(new_lp - batch["old_log_prob"][idx])  # pi_new / pi_old (PPO probability ratio)
            adv = batch["advantage"][idx]
            surr1 = ratio * adv                            # unclipped surrogate
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv  # clipped surrogate (trust region)
            policy_loss = -torch.min(surr1, surr2).mean()  # pessimistic min => clipped-PPO objective (maximize => negate)
            value_loss = 0.5 * (batch["ret"][idx] - value).pow(2).mean()  # 1/2 MSE critic regression to GAE returns
            entropy_loss = entropy.mean()                  # mean policy entropy (exploration bonus term)

            # Joint loss: maximize surrogate + entropy, minimize value error.
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(lower.parameters(), max_grad_norm)  # clip grads before stepping
            optimizer.step()

            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy_loss.detach().cpu()))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}  # average each metric over all steps


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Standardize a 1-D advantage tensor to zero mean and unit variance.

    Per-batch advantage normalization is the standard PPO variance-reduction trick: it
    keeps the surrogate-objective gradient scale stable across updates regardless of the
    reward magnitude of the active paradigm (reciprocal-distance vs. velocity-dot in
    stage 1, §5.2.1), so a single learning rate works throughout training.

    Args:
        x (torch.Tensor): [n] flat advantage values (typically the GAE output).

    Returns:
        torch.Tensor: [n] the same advantages shifted by their mean and scaled by their
        (biased) std; ``1e-8`` is added to the denominator to avoid divide-by-zero.
    """
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)  # biased std + eps for numerical safety
