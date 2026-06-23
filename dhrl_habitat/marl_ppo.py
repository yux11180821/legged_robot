# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""CTDE multi-agent PPO core for stage-2 (centralized training, decentralized execution).

Design (homogeneous agents, paper Eq. 1):
  * ONE shared actor applied per-agent on its own exo obs (decentralized execution);
  * ONE centralized critic on the joint observation (training only);
  * ONE team reward per env-step; team advantage (from the centralized critic)
    is broadcast to every agent's command log-prob -- standard parameter-shared
    MAPPO with a shared reward.

GAE is the truncation-aware version (reused from habitat_commander.ppo): the
time-limit truncation bootstraps V(terminal); task success is a true terminal.

Pure numpy/torch; unit-tested in selftest.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from habitat_commander.ppo import compute_gae_truncated  # noqa: E402  (re-exported)

__all__ = ["compute_gae_truncated", "MARLBatch", "marl_ppo_update", "broadcast_team_advantage"]


@dataclass
class MARLBatch:
    """Flattened stage-2 rollout.

    Actor samples are per (step, env, agent): S = T * n_envs * n_agents.
    Critic samples are per (step, env):       E = T * n_envs.
    """

    actor_obs: torch.Tensor          # [S, exo_dim]
    actor_hidden: torch.Tensor | None  # [S, H] stored GRU input states (None if memoryless)
    raw_actions: torch.Tensor        # [S, act_dim] pre-tanh actions
    old_log_probs: torch.Tensor      # [S]
    advantages: torch.Tensor         # [S] team advantage broadcast per agent
    critic_obs: torch.Tensor         # [E, joint_dim]
    returns: torch.Tensor            # [E] team returns


def broadcast_team_advantage(team_adv: np.ndarray, n_agents: int) -> np.ndarray:
    """[T, n_envs] team advantage -> [T, n_envs, n_agents] (same value per agent)."""
    return np.repeat(team_adv[:, :, None], n_agents, axis=2)


def marl_ppo_update(
    actor: nn.Module,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: MARLBatch,
    *,
    clip_param: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    num_minibatches: int = 4,
    target_kl: float | None = 0.02,
) -> dict[str, float]:
    """Clipped-PPO update over actor (per-agent samples) + centralized critic.

    Actor and critic minibatches are drawn with independent permutations of
    their own sample axes (S vs E) and stepped together; the two loss terms are
    independent, so pairing is arbitrary but keeps one optimizer step per
    minibatch.
    """
    S = batch.actor_obs.shape[0]
    E = batch.critic_obs.shape[0]
    k = max(1, num_minibatches)
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
    stop_early = False  # KL early-stop guard against destructive late updates

    for _ in range(ppo_epochs):
        if stop_early:
            break
        # torch.chunk keeps remainder samples (no silent drop when S % k != 0)
        chunks_s = torch.chunk(torch.randperm(S, device=batch.actor_obs.device), k)
        chunks_e = torch.chunk(torch.randperm(E, device=batch.critic_obs.device), k)
        for idx_s, idx_e in zip(chunks_s, chunks_e):
            if idx_s.numel() == 0 or idx_e.numel() == 0:
                continue

            hid = None if batch.actor_hidden is None else batch.actor_hidden[idx_s]
            new_lp, entropy = actor.evaluate_actions(
                batch.actor_obs[idx_s], hid, batch.raw_actions[idx_s]
            )
            ratio = torch.exp(new_lp - batch.old_log_probs[idx_s])
            adv = batch.advantages[idx_s]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = entropy.mean()

            values = critic(batch.critic_obs[idx_e])
            value_loss = 0.5 * (batch.returns[idx_e] - values).pow(2).mean()

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], max_grad_norm
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch.old_log_probs[idx_s] - new_lp).mean()
            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy_loss.detach().cpu()))
            kl = float(approx_kl.detach().cpu())
            stats["approx_kl"].append(kl)
            if target_kl is not None and kl > target_kl:
                stop_early = True  # this update moved the policy too far; stop further epochs
                break

    return {key: float(np.mean(vals)) for key, vals in stats.items()}
