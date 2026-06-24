# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Per-agent 3-layer HRL ACTOR (paper Fig.2):

    e_t ─▶ UL(感知) ─feature─▶ ML(RNN, 记忆 h_t) ─command─▶ frozen LL(p_t, command) ─▶ a_t

The PPO action variable is the COMMAND emitted by the ML (UL+ML are trained, the LL
is frozen), so ``act`` returns the pre-tanh command for the importance ratio.  One
instance is shared across homogeneous agents and run FULLY DECENTRALIZED -- each
agent sees only its own (e_t, p_t).  The value function is a SEPARATE decentralized
critic (``networks/critic.py``), per IPPO; this module is the actor only.
"""
from __future__ import annotations

import torch
from torch import nn

from .distributions import TanhNormal
from .lower_layer import LowerLayer
from .middle_layer import MiddleLayer
from .upper_layer import UpperLayer


class HRLPolicy(nn.Module):
    def __init__(self, exo_dim: int, proprio_dim: int, command_dim: int, action_dim: int,
                 hidden_size: int = 256, use_memory: bool = True, lower: LowerLayer | None = None):
        super().__init__()
        self.upper = UpperLayer(exo_dim, hidden_size)
        self.middle = MiddleLayer(self.upper.feature_dim, hidden_size, command_dim, use_memory)
        self.lower = lower or LowerLayer(proprio_dim, command_dim, action_dim, hidden_size)
        self.lower.freeze()   # stage-2: LL never trained; its grads are off

    def initial_hidden(self, batch_size: int, device):
        return self.middle.initial_hidden(batch_size, device)

    def trainable_parameters(self):
        """UL + ML params only (the frozen LL is excluded)."""
        return (p for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def act(self, exo, proprio, hidden):
        """Rollout step -> (raw_command, command, env_action, log_prob, next_hidden)."""
        feature = self.upper(exo)
        cmd_dist, next_hidden = self.middle(feature, hidden)
        raw_command = cmd_dist.sample_raw()
        command = TanhNormal.to_env(raw_command)
        log_prob = cmd_dist.log_prob_from_raw(raw_command)
        env_action = self.lower.action(proprio, command)
        return raw_command, command, env_action, log_prob, next_hidden

    def evaluate_actions(self, exo, hidden, raw_command):
        """PPO update over UL+ML -> (log_prob, entropy) of the stored raw command."""
        feature = self.upper(exo)
        cmd_dist, _ = self.middle(feature, hidden)
        return cmd_dist.log_prob_from_raw(raw_command), cmd_dist.entropy()

    @torch.no_grad()
    def act_deterministic(self, exo, proprio, hidden):
        """EVAL with the network LOCKED: command = tanh(mean), no sampling."""
        feature = self.upper(exo)
        cmd_dist, next_hidden = self.middle(feature, hidden)
        command = TanhNormal.deterministic(cmd_dist.base.mean)
        env_action = self.lower.action(proprio, command)
        return env_action, next_hidden
