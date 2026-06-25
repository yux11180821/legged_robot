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
    """Per-agent 3-layer HRL actor wiring UL -> ML -> frozen LL (Fig.2).

    Composes the three layers into one decentralized actor: the Upper Layer perceives
    e_t, the Middle Layer's recurrent memory turns that into a command distribution,
    and the frozen Lower Layer maps (p_t, command) to the executed action a_t. Because
    the LL is frozen in stage 2, the PPO/IPPO action variable is the ML's COMMAND, so
    the rollout path exposes the pre-tanh command for a consistent importance ratio.
    A single instance is parameter-shared across homogeneous agents and run FULLY
    DECENTRALIZED -- each agent consumes only its own (e_t, p_t) and recurrent state,
    with no global-state concatenation. The value function is a SEPARATE decentralized
    critic (``networks/critic.py``) per IPPO; this module is the actor only.

    Args:
        exo_dim: int -- exteroceptive obs e_t width (UL input).
        proprio_dim: int -- proprioceptive obs p_t width (LL input).
        command_dim: int -- command width emitted by the ML / consumed by the LL.
        action_dim: int -- low-level action a_t width sent to the env.
        hidden_size: int -- shared hidden width for UL, ML and LL (default 256).
        use_memory: bool -- enable the ML GRU; ``False`` selects the no-memory ablation.
        lower: LowerLayer | None -- a pre-trained LL to plug in; if ``None`` a fresh LL
            is created (mainly for tests, since the real pipeline injects a stage-1 LL).
    """

    def __init__(self, exo_dim: int, proprio_dim: int, command_dim: int, action_dim: int,
                 hidden_size: int = 256, use_memory: bool = True, lower: LowerLayer | None = None):
        super().__init__()
        self.upper = UpperLayer(exo_dim, hidden_size)
        # ML reads the UL feature width so the two stay dimensionally aligned.
        self.middle = MiddleLayer(self.upper.feature_dim, hidden_size, command_dim, use_memory)
        # Reuse an injected pre-trained operator, or build a fresh one when absent.
        self.lower = lower or LowerLayer(proprio_dim, command_dim, action_dim, hidden_size)
        self.lower.freeze()   # stage-2: LL never trained; its grads are off

    def initial_hidden(self, batch_size: int, device):
        """Allocate the ML's initial recurrent state h_0 for a new rollout.

        Thin delegate to the Middle Layer so callers start each episode from a clean
        spatiotemporal memory.

        Args:
            batch_size: int -- number of parallel env instances (sequences).
            device: torch device for the hidden tensor.

        Returns:
            [batch_size, hidden_size] zero state, or ``None`` in the no-memory ablation.
        """
        return self.middle.initial_hidden(batch_size, device)

    def trainable_parameters(self):
        """Yield UL + ML parameters only (the frozen LL is excluded).

        The IPPO optimizer is built from this generator so the stage-2 gradient update
        touches only the trained layers and never the frozen locomotion operator.

        Returns:
            generator over ``nn.Parameter`` with ``requires_grad=True`` (UL and ML).
        """
        return (p for p in self.parameters() if p.requires_grad)  # LL params have grads off

    @torch.no_grad()
    def act(self, exo, proprio, hidden):
        """Rollout step -> (raw_command, command, env_action, log_prob, next_hidden).

        The stage-2 data-collection forward pass through the full stack (Fig.2): UL
        perceives e_t, the ML samples a command, and the frozen LL turns it into the
        executed action. Returns the PRE-tanh command (the PPO action variable) so the
        update can re-score it consistently, plus the squashed command, the env action,
        its log-prob, and the advanced recurrent state.

        Args:
            exo: [B, exo_dim] exteroceptive observation e_t.
            proprio: [B, proprio_dim] proprioceptive observation p_t.
            hidden: [B, hidden_size] previous ML state, or ``None`` to start fresh /
                in the no-memory ablation.

        Returns:
            tuple ``(raw_command, command, env_action, log_prob, next_hidden)`` with
            shapes [B, command_dim], [B, command_dim] in (-1, 1), [B, action_dim] in
            (-1, 1), [B], and [B, hidden_size] (or ``None``) respectively.
        """
        feature = self.upper(exo)                                # UL: perceive e_t
        cmd_dist, next_hidden = self.middle(feature, hidden)     # ML: memory + command dist
        raw_command = cmd_dist.sample_raw()                      # pre-tanh PPO action variable
        command = TanhNormal.to_env(raw_command)                 # squash to the LL's input range
        log_prob = cmd_dist.log_prob_from_raw(raw_command)       # log-prob for the IPPO ratio
        env_action = self.lower.action(proprio, command)         # frozen LL: command -> a_t
        return raw_command, command, env_action, log_prob, next_hidden

    def evaluate_actions(self, exo, hidden, raw_command):
        """PPO update over UL+ML -> (log_prob, entropy) of the stored raw command.

        The update-time counterpart to :meth:`act` (gradients ON): re-runs UL+ML on the
        batch and scores the previously sampled raw command so IPPO can form the clipped
        surrogate and entropy-bonus terms. The LL is not invoked here -- only the trained
        layers participate in the gradient.

        Args:
            exo: [B, exo_dim] exteroceptive observation e_t from the batch.
            hidden: [B, hidden_size] ML state for this timestep (or ``None``).
            raw_command: [B, command_dim] pre-tanh command stored during :meth:`act`.

        Returns:
            tuple ``(log_prob, entropy)`` each of shape [B] -- the new log-density and
            entropy proxy of the stored command under the current UL+ML.
        """
        feature = self.upper(exo)
        cmd_dist, _ = self.middle(feature, hidden)  # discard next_hidden: scoring only
        return cmd_dist.log_prob_from_raw(raw_command), cmd_dist.entropy()

    @torch.no_grad()
    def act_deterministic(self, exo, proprio, hidden):
        """EVAL with the network LOCKED: command = tanh(mean), no sampling.

        The greedy rollout used at evaluation/metrics time: takes the ML command mean
        instead of sampling, squashes it, and runs it through the frozen LL to get the
        executed action -- giving reproducible behaviour for success-rate / inference-
        time measurement.

        Args:
            exo: [B, exo_dim] exteroceptive observation e_t.
            proprio: [B, proprio_dim] proprioceptive observation p_t.
            hidden: [B, hidden_size] previous ML state, or ``None``.

        Returns:
            tuple ``(env_action, next_hidden)`` with shapes [B, action_dim] in (-1, 1)
            and [B, hidden_size] (or ``None``).
        """
        feature = self.upper(exo)
        cmd_dist, next_hidden = self.middle(feature, hidden)
        command = TanhNormal.deterministic(cmd_dist.base.mean)  # greedy command, no sampling
        env_action = self.lower.action(proprio, command)        # frozen LL: command -> a_t
        return env_action, next_hidden
