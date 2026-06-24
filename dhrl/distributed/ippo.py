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
    """Independent PPO trainer -- the paper's fully decentralized learner (§5).

    Wraps a parameter-shared HRL actor (``policy``: the trainable UL+ML over the frozen
    LL, §4.2/§4.3) and a per-agent ``DecentralizedCritic`` and optimises BOTH with a single
    Adam over the clipped-PPO surrogate.  "Independent" means every (env, agent) trajectory
    is treated as its own single-agent PPO problem on that agent's OWN observation only --
    no global state, no centralized critic -- which is precisely the decentralization that
    lets the method scale and transfer zero-shot to more agents (§5.4, vs. the MAPPO baseline
    of Fig.5 b/c).  Homogeneous agents share these parameters and a shared reward (Eq.1), so
    all (env, agent) samples are pooled into one batch for the update.

    Args:
        policy: the shared ``HRLPolicy`` actor; must expose ``act`` / ``evaluate_actions`` /
            ``trainable_parameters`` (the LL is frozen, so only UL+ML params are returned).
        critic: a ``DecentralizedCritic`` mapping an agent's own exteroceptive obs -> scalar V.
        lr (float): Adam learning rate for the joint actor+critic parameter set.
        clip_param (float): PPO surrogate clip epsilon (the 1+-eps / 1--eps ratio band).
        value_loss_coef (float): weight on the critic MSE term in the total loss.
        entropy_coef (float): weight on the entropy bonus (0.0 -> no exploration bonus).
        max_grad_norm (float): global-norm gradient clip applied before each optimizer step.
        ppo_epochs (int): number of passes over the rollout buffer per ``update``.
        num_minibatches (int): how many minibatches each epoch is split into.
        target_kl (float | None): if set, stop the update early once the approximate KL
            between old and new policy exceeds this trust-region threshold.
    """

    def __init__(self, policy, critic, *, lr: float = 5e-4, clip_param: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.0,
                 max_grad_norm: float = 0.5, ppo_epochs: int = 4, num_minibatches: int = 4,
                 target_kl: float | None = None):
        # Store the shared actor and the per-agent (decentralized) critic.
        self.policy = policy
        self.critic = critic
        # Jointly optimise the trainable actor params (UL+ML; LL is frozen) and ALL critic params.
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
        """Sample one decentralized rollout step for every (env, agent) in parallel.

        Each agent acts on its OWN observation only: the exteroceptive ``exo`` (e_t =
        env obs + nearest-neighbour relative pose, §4.2/Fig.3) feeds UL+ML to emit the ML
        command, and the proprioceptive ``proprio`` (p_t) + that command feed the frozen LL
        to produce the env action a_t.  The PPO action variable is the ML command, so we
        also return its log-prob and the decentralized value V(exo) used later for GAE.
        Flattening (E, N) -> E*N treats every (env, agent) as an independent single-agent
        sample, which is what "Independent PPO" means.

        Args:
            exo: exteroceptive obs e_t, shape [E, N, exo_dim] (E envs, N agents).
            proprio: proprioceptive obs p_t for the frozen LL, shape [E, N, proprio_dim].
            hidden: ML recurrent state h_t fed back into the GRU, shape [E, N, H] or None
                (None when the ML memory is disabled).

        Returns:
            dict of per-(env, agent) tensors:
              ``raw_command`` [E, N, cmd_dim]   pre-squash ML command (stored for re-eval),
              ``command``     [E, N, cmd_dim]   the actual command handed to the LL,
              ``env_action``  [E, N, act_dim]   the LL's env action a_t,
              ``log_prob``    [E, N]            log pi(command|e_t) under the current policy,
              ``value``       [E, N]            V(e_t) from the decentralized critic,
              ``next_hidden`` [E, N, H] | None  updated ML hidden state for the next step.
        """
        E, N = exo.shape[:2]
        # Flatten (E, N) -> E*N so the shared net processes all agents as one batch.
        exo_f = exo.reshape(E * N, -1)
        pro_f = proprio.reshape(E * N, -1)
        hid_f = None if hidden is None else hidden.reshape(E * N, -1)
        # Actor forward: UL+ML emit the command; the frozen LL maps (p_t, command) -> a_t.
        raw, command, env_action, log_prob, next_hid = self.policy.act(exo_f, pro_f, hid_f)
        # Critic forward on the agent's OWN obs only (decentralized value, no global state).
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
        """Evaluate the decentralized critic on the post-rollout observation.

        Provides ``last_value`` = V(e_{T}) for the truncation-aware GAE bootstrap at the end
        of a rollout (the value beyond the last stored step).  Still per-agent and on the
        agent's own obs only -- no global state, consistent with the IPPO design (§4.2).

        Args:
            exo: exteroceptive obs after the final rollout step, shape [E, N, exo_dim].

        Returns:
            V(e_t) per (env, agent), shape [E, N].
        """
        E, N = exo.shape[:2]
        # Flatten -> critic -> reshape back to per-(env, agent) values.
        return self.critic(exo.reshape(E * N, -1)).reshape(E, N)

    # ------------------------------------------------------------------ #
    def update(self, buffer) -> dict[str, float]:
        """Run the clipped-PPO optimisation over a finished rollout buffer.

        This is the core of the paper's IPPO learner: it performs ``ppo_epochs`` passes over
        the pooled (T*E*N) samples, each split into ``num_minibatches`` minibatches, and on
        each minibatch minimises the clipped surrogate + value MSE - entropy bonus.  Because
        agents are homogeneous and parameter-shared (Eq.1), all (env, agent) samples train the
        same actor/critic; the critic is still decentralized (V on the agent's own obs).  An
        optional KL trust-region check early-stops the update if the policy moves too far.

        Args:
            buffer: a ``RolloutBuffer`` whose ``compute_returns`` has already populated GAE
                advantages and returns; it yields minibatch dicts with keys exo/hidden/
                raw_command/old_log_prob/advantage/ret.

        Returns:
            dict of epoch-averaged scalars: ``policy_loss``, ``value_loss``, ``entropy``,
            ``approx_kl`` (each a Python float, 0.0 if no minibatch was processed).
        """
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        stop = False  # set by the KL early-stop to break out of the outer epoch loop
        for _ in range(self.ppo_epochs):
            if stop:
                break
            for mb in buffer.minibatches(self.num_minibatches):
                # Re-evaluate the stored commands under the CURRENT policy to get fresh log-probs.
                new_lp, entropy = self.policy.evaluate_actions(
                    mb["exo"], mb["hidden"], mb["raw_command"])
                # PPO importance ratio pi_new / pi_old = exp(log pi_new - log pi_old).
                ratio = torch.exp(new_lp - mb["old_log_prob"])
                adv = mb["advantage"]
                surr1 = ratio * adv                                                   # unclipped surrogate
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv  # clipped
                policy_loss = -torch.min(surr1, surr2).mean()  # pessimistic min -> maximise lower bound

                # Decentralized critic regression toward the GAE returns (own-obs value target).
                value = self.critic(mb["exo"])
                value_loss = 0.5 * (mb["ret"] - value).pow(2).mean()
                entropy_loss = entropy.mean()  # mean policy entropy (exploration bonus term)

                # Total loss: clipped policy + weighted value MSE - entropy bonus.
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)  # stabilise with global-norm clip
                self.optimizer.step()

                with torch.no_grad():
                    # Cheap KL estimate (old - new) used only for monitoring / early-stop.
                    approx_kl = (mb["old_log_prob"] - new_lp).mean()
                stats["policy_loss"].append(float(policy_loss.detach().cpu()))
                stats["value_loss"].append(float(value_loss.detach().cpu()))
                stats["entropy"].append(float(entropy_loss.detach().cpu()))
                stats["approx_kl"].append(float(approx_kl.detach().cpu()))
                if self.target_kl is not None and float(approx_kl) > self.target_kl:
                    stop = True  # KL early-stop: this update moved the policy too far
                    break
        # Average each tracked quantity across all minibatches/epochs for logging.
        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
