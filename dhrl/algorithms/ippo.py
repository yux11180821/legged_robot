# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""IPPO -- Independent PPO (THE PAPER'S METHOD, §5; fully decentralized).

Self-contained learner (MARL_for_MeltingPot style): given ``env_info`` + ``args`` it
builds the shared 3-layer HRL actor (UL+ML over a FROZEN lower operator), a per-agent
``DecentralizedCritic``, and one Adam optimiser; it exposes ``init_hidden`` /
``take_action`` / ``update`` / ``save``.  Every (env, agent) is its own single-agent
PPO problem on its OWN observation -- no centralized critic, no global state -- which is
the decentralization that lets the method scale (§4.2/§5.4; vs the MAPPO baseline of
Fig.5 b/c).  GAE (truncation-aware) and the clipped-PPO update both live in ``update``.
"""
import os

import numpy as np
import torch
from torch import nn

from networks.critic import DecentralizedCritic
from networks.hrl_policy import HRLPolicy
from networks.lower_layer import LowerLayer


class IPPO:
    def __init__(self, env_info, args):
        """Build the actor (UL+ML+frozen LL), the decentralized critic and the optimiser.

        Args:
            env_info (dict): n_agents / exo_dim / proprio_dim / command_dim / action_dim.
            args: flat config (hidden_dim, lr, clip, gamma, gae_lambda, ppo_epochs,
                num_minibatches, entropy_coef, target_kl, use_memory, lower_ckpt, device).
        """
        self.n_agents = env_info["n_agents"]
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.hidden_dim = getattr(args, "hidden_dim", 256)
        self.use_memory = getattr(args, "use_memory", True)  # False -> no-memory ablation

        exo, pro = env_info["exo_dim"], env_info["proprio_dim"]
        cmd, act = env_info["command_dim"], env_info["action_dim"]

        # FROZEN lower operator (stage-1 product); a fresh frozen LL is fine for a smoke.
        ckpt = getattr(args, "lower_ckpt", None)
        if ckpt and os.path.exists(ckpt):
            lower = LowerLayer.load_frozen(ckpt, proprio_dim=pro, command_dim=cmd,
                                           action_dim=act, hidden_size=self.hidden_dim)
        else:
            lower = LowerLayer(pro, cmd, act, self.hidden_dim).freeze()
        self.actor = HRLPolicy(exo, pro, cmd, act, self.hidden_dim,
                               use_memory=self.use_memory, lower=lower).to(self.device)
        self.critic = DecentralizedCritic(exo, self.hidden_dim).to(self.device)

        # Optimise UL+ML (trainable_parameters excludes the frozen LL) + the critic.
        self._params = list(self.actor.trainable_parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self._params, lr=getattr(args, "lr", 5e-4), eps=1e-5)

        # PPO hyper-parameters (paper §5).
        self.gamma = getattr(args, "gamma", 0.995)
        self.gae_lambda = getattr(args, "gae_lambda", 0.95)
        self.clip = getattr(args, "clip", 0.2)
        self.value_coef = getattr(args, "value_coef", 0.5)
        self.entropy_coef = getattr(args, "entropy_coef", 1e-3)
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        self.ppo_epochs = getattr(args, "ppo_epochs", 4)
        self.num_minibatches = getattr(args, "num_minibatches", 4)
        self.target_kl = getattr(args, "target_kl", None)

    def init_hidden(self, num_envs):
        """Initial ML hidden state, shape [num_envs, n_agents, hidden_dim] (None if no memory)."""
        if not self.use_memory:
            return None
        return self.actor.initial_hidden(num_envs * self.n_agents, self.device).view(
            num_envs, self.n_agents, self.hidden_dim)

    @torch.no_grad()
    def take_action(self, exo, proprio, hidden):
        """Decentralized rollout step for all (env, agent).  exo/proprio: [E, N, *] tensors.

        Returns a dict of [E, N, *] tensors: raw_command (PPO action var), env_action (a_t
        applied to the sim), log_prob, value (own-obs critic), next_hidden.
        """
        E, N = exo.shape[:2]
        exo_f, pro_f = exo.reshape(E * N, -1), proprio.reshape(E * N, -1)
        hid_f = None if hidden is None else hidden.reshape(E * N, -1)
        raw, command, env_action, log_prob, next_hid = self.actor.act(exo_f, pro_f, hid_f)
        value = self.critic(exo_f)  # decentralized: value of the agent's OWN obs only
        return {
            "raw_command": raw.reshape(E, N, -1),
            "env_action": env_action.reshape(E, N, -1),
            "log_prob": log_prob.reshape(E, N),
            "value": value.reshape(E, N),
            "next_hidden": None if next_hid is None else next_hid.reshape(E, N, -1),
        }

    @torch.no_grad()
    def get_value(self, exo):
        """V(e_t) per (env, agent) -> [E, N]; used as the GAE bootstrap after the rollout."""
        E, N = exo.shape[:2]
        return self.critic(exo.reshape(E * N, -1)).reshape(E, N)

    def update(self, buffer, last_value):
        """Truncation-aware GAE + clipped-PPO over a finished rollout buffer.

        Args:
            buffer: a filled RolloutBuffer.
            last_value: V of the obs after the last rollout step, [E, N], for the bootstrap.
        Returns:
            dict of averaged scalars (policy_loss, value_loss, entropy, approx_kl).
        """
        buffer.compute_returns(last_value, self.gamma, self.gae_lambda)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        stop = False
        for _ in range(self.ppo_epochs):
            if stop:
                break
            for mb in buffer.minibatches(self.num_minibatches):
                new_lp, entropy = self.actor.evaluate_actions(mb["exo"], mb["hidden"], mb["raw_command"])
                ratio = torch.exp(new_lp - mb["old_log_prob"])
                adv = mb["advantage"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb["ret"] - self.critic(mb["exo"])).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float((mb["old_log_prob"] - new_lp).mean())
                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(entropy.mean()))
                stats["approx_kl"].append(approx_kl)
                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop = True  # KL trust-region early stop
                    break
        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    def save(self, path):
        """Save actor + critic state dicts."""
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        """Load actor + critic state dicts."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
