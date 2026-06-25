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
    """Compute Generalized Advantage Estimation with correct time-limit (truncation) handling.

    Standard GAE assumes every ``done`` is a true terminal whose future value is 0, which is
    wrong for episodes that end only because a time limit was hit -- there the trajectory was
    artificially cut and the future return should still be bootstrapped from V(terminal_obs).
    This routine therefore distinguishes the two: on a done step it bootstraps the next value
    from ``bootstrap[t]`` (= V(terminal_obs) for a truncation, 0 for a real terminal), while the
    (1 - done) factor still severs the GAE accumulation so advantage never bleeds across an
    episode boundary.  Correct truncation bootstrapping is what keeps the IPPO returns unbiased
    on the cooperation tasks (§5.2). Operates on each flattened (env, agent) column independently.

    Args:
        rewards: per-step shared rewards r_t (Eq.1), shape [T, B] (B = E*N flattened).
        values: critic estimates V(s_t) at each stored step, shape [T, B].
        dones: episode-boundary flags (1 at the last step of an episode), shape [T, B].
        bootstrap: V(terminal_obs) on a time-limit truncation else 0, shape [T, B].
        last_value: V of the obs AFTER the final stored step, shape [B] (end-of-rollout bootstrap).
        gamma (float): discount factor.
        gae_lambda (float): GAE lambda trace-decay (bias/variance trade-off).

    Returns:
        (advantages [T, B], returns [T, B]) where returns = advantages + values.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)  # running GAE accumulator
    for t in reversed(range(T)):  # backward recursion over time
        # Bootstrap value of the next state: end-of-rollout uses last_value, else the stored V_{t+1}.
        next_value = last_value if t == T - 1 else values[t + 1]
        # On an episode boundary, replace next value by the truncation bootstrap (V(term) or 0).
        next_value = torch.where(dones[t] > 0.5, bootstrap[t], next_value)
        delta = rewards[t] + gamma * next_value - values[t]  # TD residual delta_t
        # Accumulate GAE; (1 - done) zeroes the trace so advantage does not cross episode resets.
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        adv[t] = last_gae
    return adv, adv + values  # value targets (returns) = advantage + baseline


class RolloutBuffer:
    """Fixed-size on-policy storage for a T-step rollout across N agents in E envs.

    Holds exactly the quantities the IPPO update needs and nothing else.  Only ``exo`` (e_t) and
    the ML ``hidden`` are stored for the actor, because the trainable part is UL+ML which see only
    the exteroceptive obs and the GRU memory (§4.2/§4.3) -- the proprioceptive p_t fed the FROZEN
    LL at rollout time and is never re-evaluated, so it is deliberately not kept.  Every (env,
    agent) column is an independent trajectory for GAE; because homogeneous agents share params
    (Eq.1) they are pooled into one batch at update time.

    Args:
        rollout_steps (int): T, number of steps stored per rollout.
        n_envs (int): E, number of parallel environments.
        n_agents (int): N, number of agents per environment.
        exo_dim (int): dimension of the exteroceptive obs e_t stored per (env, agent).
        command_dim (int): dimension of the ML raw command (the PPO action variable).
        hidden_size (int): dimension of the ML GRU hidden state (ignored if no memory).
        device: torch device on which all buffers are allocated.
        use_memory (bool): whether the ML recurrent hidden state is stored/used.
    """

    def __init__(self, rollout_steps: int, n_envs: int, n_agents: int, exo_dim: int,
                 command_dim: int, hidden_size: int, device, use_memory: bool = True):
        self.T, self.E, self.N = rollout_steps, n_envs, n_agents
        self.device = device
        self.use_memory = use_memory
        sh = (self.T, n_envs, n_agents)  # common [T, E, N] prefix for all per-agent tensors
        self.exo = torch.zeros(*sh, exo_dim, device=device)                                # e_t
        self.hidden = torch.zeros(*sh, hidden_size, device=device) if use_memory else None  # ML h_t
        self.raw_command = torch.zeros(*sh, command_dim, device=device)                    # PPO action
        self.log_prob = torch.zeros(*sh, device=device)                                    # log pi(a|e)
        self.value = torch.zeros(*sh, device=device)                                       # V(e_t)
        self.reward = torch.zeros(*sh, device=device)                                      # shared r_t
        self.bootstrap = torch.zeros(*sh, device=device)                                   # V(term) on trunc.
        self.done = torch.zeros(self.T, n_envs, device=device)  # episode boundary is per-ENV (shared by its agents)
        self.step = 0                       # write cursor into the time dimension
        self.advantages = self.returns = None  # filled in by compute_returns

    def insert(self, *, exo, hidden, raw_command, log_prob, value, reward, done, bootstrap):
        """Write one rollout step's data at the current time cursor and advance it.

        ``hidden`` is deliberately the INPUT hidden state that produced this step's action (not
        ``next_hidden``), because ``evaluate_actions`` must re-run the ML/GRU from the same input
        memory to recover gradient-carrying log-probs during the update.

        Args:
            exo: exteroceptive obs e_t, shape [E, N, exo_dim].
            hidden: ML input hidden state h_t for this step, shape [E, N, H] (used iff use_memory).
            raw_command: sampled raw ML command (PPO action), shape [E, N, command_dim].
            log_prob: log pi(command|e_t) under the acting policy, shape [E, N].
            value: critic value V(e_t), shape [E, N].
            reward: shared per-step reward r_t (Eq.1), shape [E, N].
            done: episode-boundary flag per env, shape [E] (broadcast to that env's N agents).
            bootstrap: V(terminal_obs) on a time-limit truncation else 0, shape [E, N].

        Returns:
            None (mutates the buffer in place and increments the step cursor).
        """
        t = self.step
        self.exo[t] = exo
        if self.use_memory:
            self.hidden[t] = hidden  # only meaningful when the ML memory is enabled
        self.raw_command[t] = raw_command
        self.log_prob[t] = log_prob
        self.value[t] = value
        self.reward[t] = reward
        self.done[t] = done  # [E]: one boundary flag per env, shared by all its agents
        self.bootstrap[t] = bootstrap
        self.step += 1  # advance the time-dimension write cursor

    def compute_returns(self, last_value, gamma: float, gae_lambda: float):
        """Fill ``self.advantages`` / ``self.returns`` via truncation-aware GAE over the rollout.

        Flattens the per-env done flag to per-(env, agent) and folds (E, N) into one batch axis so
        each (env, agent) trajectory is advantaged independently, then delegates the recursion to
        ``compute_gae_truncated``.  Must be called once a rollout is complete and before sampling
        minibatches for the IPPO update.

        Args:
            last_value: V of the obs after the final stored step, shape [E, N] (GAE bootstrap).
            gamma (float): discount factor.
            gae_lambda (float): GAE lambda trace-decay.

        Returns:
            None (stores advantages [T, E, N] and returns [T, E, N] on the buffer).
        """
        T, E, N = self.T, self.E, self.N
        # Broadcast the per-env done [T, E] to per-agent [T, E, N], then flatten agents into the batch.
        done = self.done.unsqueeze(-1).expand(T, E, N).reshape(T, E * N)
        adv, ret = compute_gae_truncated(
            self.reward.reshape(T, E * N), self.value.reshape(T, E * N), done,
            self.bootstrap.reshape(T, E * N), last_value.reshape(E * N), gamma, gae_lambda)
        # Restore the [T, E, N] layout for storage.
        self.advantages = adv.reshape(T, E, N)
        self.returns = ret.reshape(T, E, N)

    def minibatches(self, num_minibatches: int, normalize_adv: bool = True):
        """Yield randomly shuffled minibatches over all pooled (T*E*N) samples.

        All time / env / agent dimensions are flattened into one sample axis because the
        homogeneous agents share parameters (Eq.1), so every (t, env, agent) sample trains the
        same actor/critic.  Advantages are (optionally) standardized across the whole rollout
        for stable PPO gradients, and the index permutation decorrelates consecutive samples.

        Args:
            num_minibatches (int): number of (roughly equal) chunks to split the rollout into.
            normalize_adv (bool): if True, z-score the advantages over the whole batch.

        Yields:
            dict per minibatch with: ``exo`` [b, exo_dim], ``hidden`` [b, H] or None,
            ``raw_command`` [b, command_dim], ``old_log_prob`` [b], ``advantage`` [b], ``ret`` [b]
            (b = that chunk's sample count).
        """
        S = self.T * self.E * self.N  # total pooled samples
        exo = self.exo.reshape(S, -1)
        hidden = self.hidden.reshape(S, -1) if self.use_memory else None
        raw = self.raw_command.reshape(S, -1)
        old_lp = self.log_prob.reshape(S)
        adv = self.advantages.reshape(S)
        ret = self.returns.reshape(S)
        if normalize_adv:
            # Standardize advantages (zero mean, unit std) for PPO gradient stability.
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        # Random permutation split into chunks -> shuffled minibatches.
        for idx in torch.randperm(S, device=self.device).chunk(max(1, num_minibatches)):
            if idx.numel() == 0:
                continue  # skip an empty chunk (can happen if S < num_minibatches)
            yield {
                "exo": exo[idx],
                "hidden": None if hidden is None else hidden[idx],
                "raw_command": raw[idx],
                "old_log_prob": old_lp[idx],
                "advantage": adv[idx],
                "ret": ret[idx],
            }

    def reset(self):
        """Rewind the write cursor so the buffer can be refilled by the next rollout.

        Only the cursor is reset; the underlying tensors are overwritten in place on the next
        ``insert`` calls, so no reallocation is needed between rollouts.

        Returns:
            None (resets the step cursor to 0).
        """
        self.step = 0
