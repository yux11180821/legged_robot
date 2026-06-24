# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""On-policy rollout storage + truncation-aware GAE.

Stores a T-step rollout for N agents in E envs.  Because the trainable actor is
UL+ML (the LL is frozen) and those only see the exteroceptive obs ``e_t`` and the
ML hidden state, we store ``exo`` and ``hidden`` -- NOT proprio (it only fed the
frozen LL at rollout time).  Each (env, agent) is an independent trajectory for
GAE; homogeneous agents are pooled into one parameter-shared batch.
"""
from __future__ import annotations

import torch


def compute_gae_truncated(rewards, values, dones, bootstrap, last_value, gamma, gae_lambda):
    """Truncation-aware GAE.  All of ``rewards/values/dones/bootstrap`` are [T, B]
    (B = flattened env*agent); ``last_value`` is [B].

      * ``dones[t] == 1`` marks an episode boundary (the env was then reset);
      * ``bootstrap[t]`` = V(terminal_obs) on a TIME-LIMIT truncation, else 0 -- so a
        time-limit truncation still bootstraps while a true terminal does not;
      * the GAE recursion is cut by (1 - dones[t]) so advantage never leaks across
        an episode boundary.
    Returns (advantages [T, B], returns [T, B]).
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_value = torch.where(dones[t] > 0.5, bootstrap[t], next_value)
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        adv[t] = last_gae
    return adv, adv + values


class RolloutBuffer:
    def __init__(self, rollout_steps: int, n_envs: int, n_agents: int, exo_dim: int,
                 command_dim: int, hidden_size: int, device, use_memory: bool = True):
        self.T, self.E, self.N = rollout_steps, n_envs, n_agents
        self.device = device
        self.use_memory = use_memory
        sh = (self.T, n_envs, n_agents)
        self.exo = torch.zeros(*sh, exo_dim, device=device)
        self.hidden = torch.zeros(*sh, hidden_size, device=device) if use_memory else None
        self.raw_command = torch.zeros(*sh, command_dim, device=device)
        self.log_prob = torch.zeros(*sh, device=device)
        self.value = torch.zeros(*sh, device=device)
        self.reward = torch.zeros(*sh, device=device)
        self.bootstrap = torch.zeros(*sh, device=device)
        self.done = torch.zeros(self.T, n_envs, device=device)
        self.step = 0
        self.advantages = self.returns = None

    def insert(self, *, exo, hidden, raw_command, log_prob, value, reward, done, bootstrap):
        """Per-step tensors: exo/hidden/raw_command/log_prob/value/reward/bootstrap are
        [E, N, ...]; ``done`` is [E] (episode boundary shared by an env's agents).
        ``hidden`` is the INPUT hidden state at this step (fed back at update time)."""
        t = self.step
        self.exo[t] = exo
        if self.use_memory:
            self.hidden[t] = hidden
        self.raw_command[t] = raw_command
        self.log_prob[t] = log_prob
        self.value[t] = value
        self.reward[t] = reward
        self.done[t] = done
        self.bootstrap[t] = bootstrap
        self.step += 1

    def compute_returns(self, last_value, gamma: float, gae_lambda: float):
        """``last_value``: [E, N] value of the obs after the final rollout step."""
        T, E, N = self.T, self.E, self.N
        done = self.done.unsqueeze(-1).expand(T, E, N).reshape(T, E * N)
        adv, ret = compute_gae_truncated(
            self.reward.reshape(T, E * N), self.value.reshape(T, E * N), done,
            self.bootstrap.reshape(T, E * N), last_value.reshape(E * N), gamma, gae_lambda)
        self.advantages = adv.reshape(T, E, N)
        self.returns = ret.reshape(T, E, N)

    def minibatches(self, num_minibatches: int, normalize_adv: bool = True):
        """Yield flattened (T*E*N) minibatches for the parameter-shared update."""
        S = self.T * self.E * self.N
        exo = self.exo.reshape(S, -1)
        hidden = self.hidden.reshape(S, -1) if self.use_memory else None
        raw = self.raw_command.reshape(S, -1)
        old_lp = self.log_prob.reshape(S)
        adv = self.advantages.reshape(S)
        ret = self.returns.reshape(S)
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        for idx in torch.randperm(S, device=self.device).chunk(max(1, num_minibatches)):
            if idx.numel() == 0:
                continue
            yield {
                "exo": exo[idx],
                "hidden": None if hidden is None else hidden[idx],
                "raw_command": raw[idx],
                "old_log_prob": old_lp[idx],
                "advantage": adv[idx],
                "ret": ret[idx],
            }

    def reset(self):
        self.step = 0
