# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""IPPO -- Independent PPO (THE PAPER'S METHOD, §5; fully decentralized).

The shared ``HRLPolicy`` actor and a ``DecentralizedCritic`` are applied PER AGENT,
each on that agent's OWN observation only.  There is NO centralized critic and NO
concatenation of all agents' observations into a global state -- that (CTDE/MAPPO)
is the baseline the paper beats and the thing that makes training fail to converge
as the agent count grows (Fig.5 b/c).  Homogeneous agents share parameters and a
shared reward (Eq.1); the clipped-PPO objective + truncation-aware GAE come from the
rollout buffer.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class IPPO:
    def __init__(self, policy, critic, *, lr: float = 5e-4, clip_param: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.0,
                 max_grad_norm: float = 0.5, ppo_epochs: int = 4, num_minibatches: int = 4,
                 target_kl: float | None = None):
        self.policy = policy
        self.critic = critic
        self._params = list(policy.trainable_parameters()) + list(critic.parameters())
        self.optimizer = torch.optim.Adam(self._params, lr=lr, eps=1e-5)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.target_kl = target_kl

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, exo, proprio, hidden):
        """Decentralized rollout step.  exo/proprio: [E, N, *]; hidden: [E, N, H] | None.
        Returns a dict of [E, N, *] tensors (raw_command/command/env_action/log_prob/
        value/next_hidden)."""
        E, N = exo.shape[:2]
        exo_f = exo.reshape(E * N, -1)
        pro_f = proprio.reshape(E * N, -1)
        hid_f = None if hidden is None else hidden.reshape(E * N, -1)
        raw, command, env_action, log_prob, next_hid = self.policy.act(exo_f, pro_f, hid_f)
        value = self.critic(exo_f)
        return {
            "raw_command": raw.reshape(E, N, -1),
            "command": command.reshape(E, N, -1),
            "env_action": env_action.reshape(E, N, -1),
            "log_prob": log_prob.reshape(E, N),
            "value": value.reshape(E, N),
            "next_hidden": None if next_hid is None else next_hid.reshape(E, N, -1),
        }

    @torch.no_grad()
    def value(self, exo):
        """Decentralized value of the final obs -> [E, N] (for the GAE bootstrap)."""
        E, N = exo.shape[:2]
        return self.critic(exo.reshape(E * N, -1)).reshape(E, N)

    # ------------------------------------------------------------------ #
    def update(self, buffer) -> dict[str, float]:
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        stop = False
        for _ in range(self.ppo_epochs):
            if stop:
                break
            for mb in buffer.minibatches(self.num_minibatches):
                new_lp, entropy = self.policy.evaluate_actions(
                    mb["exo"], mb["hidden"], mb["raw_command"])
                ratio = torch.exp(new_lp - mb["old_log_prob"])
                adv = mb["advantage"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(mb["exo"])
                value_loss = 0.5 * (mb["ret"] - value).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb["old_log_prob"] - new_lp).mean()
                stats["policy_loss"].append(float(policy_loss.detach().cpu()))
                stats["value_loss"].append(float(value_loss.detach().cpu()))
                stats["entropy"].append(float(entropy_loss.detach().cpu()))
                stats["approx_kl"].append(float(approx_kl.detach().cpu()))
                if self.target_kl is not None and float(approx_kl) > self.target_kl:
                    stop = True  # KL early-stop: this update moved the policy too far
                    break
        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
