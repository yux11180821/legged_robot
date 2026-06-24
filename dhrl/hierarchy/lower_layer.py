# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Lower Layer (LL) -- pre-trained locomotion operator (paper §4.1, §5.2.1, Fig.2).

Maps (proprioceptive obs p_t, command) -> low-level action a_t.

  * STAGE 1: trained single-agent with ``ppo_single`` (it carries its own actor +
    critic heads).  Two paradigms (``mode``): 'position' (reward = reciprocal L2
    distance to a target position) and 'velocity' (reward = velocity dot product).
    The paper freezes the POSITION-mode operator.
  * STAGE 2: ``freeze()`` -> used deterministically inside the HRL policy; its
    parameters get requires_grad=False so the IPPO optimiser never touches it.
"""
from __future__ import annotations

import torch
from torch import nn

from .action import TanhNormal


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    """Build the shared two-layer ``tanh`` encoder trunk used by the Lower Layer.

    Factored out so the LL body is a single 2x ``tanh`` MLP; tiny helper, but it fixes
    the operator's encoder architecture in one place.

    Args:
        in_dim: int -- input width (here ``proprio_dim + command_dim``, the concatenated
            proprioceptive observation p_t and the command).
        hidden: int -- width of both hidden layers.

    Returns:
        ``nn.Sequential`` mapping [..., in_dim] -> [..., hidden] (a dense latent).
    """
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh())


class LowerLayer(nn.Module):
    """Lower Layer (LL) -- the pre-trained, then frozen, locomotion operator (§4.1, §5.2.1).

    Maps a proprioceptive observation p_t plus a command to a low-level action a_t.
    It is the bottom of the HRL stack and the only component shared across the two
    training stages with different roles:
      * STAGE 1 (single-agent pre-training): it owns its own actor head
        (``action_mean`` + ``log_std``) AND a ``value_head``, and is optimized by the
        single-agent PPO loop. Two reward paradigms select via ``mode``: 'position'
        (reward = reciprocal L2 distance to a target position) and 'velocity'
        (reward = velocity dot target direction); the paper freezes the POSITION-mode
        operator for the cooperation tasks.
      * STAGE 2 (cooperation): :meth:`freeze` turns off all grads and the operator is
        called DETERMINISTICALLY inside the HRL policy, so the IPPO optimizer that
        trains UL+ML never updates it.

    Args:
        proprio_dim: int -- width of the proprioceptive observation p_t.
        command_dim: int -- width of the command produced by the Middle Layer.
        action_dim: int -- width of the low-level action a_t sent to the robot.
        hidden_size: int -- encoder hidden width (default 256).
        mode: str -- reward paradigm used during stage-1 pre-training ('position' or
            'velocity'); stored so it round-trips through save/load.
        init_log_std: float -- initial log-std for every action component (default -0.5).
    """

    def __init__(self, proprio_dim: int, command_dim: int, action_dim: int,
                 hidden_size: int = 256, mode: str = "position", init_log_std: float = -0.5):
        super().__init__()
        self.mode = mode  # 'position' or 'velocity' stage-1 reward paradigm
        # Encoder consumes proprio p_t and the command concatenated together.
        self.encoder = _mlp(proprio_dim + command_dim, hidden_size)
        self.action_mean = nn.Linear(hidden_size, action_dim)            # pre-tanh action mean
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))  # learnable action log-std
        self.value_head = nn.Linear(hidden_size, 1)   # used only for stage-1 PPO

    def _latent(self, proprio, command):
        """Encode the proprioceptive obs and command into the shared LL latent.

        Concatenation + MLP that every other LL method reads from; centralizes the
        ``[p_t ; command]`` -> latent mapping.

        Args:
            proprio: [B, proprio_dim] proprioceptive observation p_t.
            command: [B, command_dim] command (env-space, in (-1, 1)) from the ML.

        Returns:
            [B, hidden_size] dense latent feeding the action and value heads.
        """
        return self.encoder(torch.cat([proprio, command], dim=-1))  # fuse p_t with the command

    def _dist(self, latent) -> TanhNormal:
        """Build the squashed-Gaussian action distribution from a latent.

        The stochastic action head used during stage-1 PPO (exploration); at stage 2
        the deterministic :meth:`action` path is used instead.

        Args:
            latent: [B, hidden_size] encoder output from :meth:`_latent`.

        Returns:
            :class:`TanhNormal` over the low-level action, shape [B, action_dim].
        """
        mean = self.action_mean(latent)
        return TanhNormal(mean, torch.exp(self.log_std).expand_as(mean))  # state-independent std

    # ----- stage 1: single-agent PPO -----
    @torch.no_grad()
    def act(self, proprio, command):
        """Sample a stage-1 rollout step (action + log-prob + value), no gradients.

        The data-collection call of the single-agent PPO loop that pre-trains the LL.
        Returns the PRE-tanh sample so the later PPO update can re-score it on the same
        variable, the squashed env action, its log-prob, and the critic's value.

        Args:
            proprio: [B, proprio_dim] proprioceptive observation p_t.
            command: [B, command_dim] command driving the operator.

        Returns:
            tuple ``(raw, env_action, log_prob, value)`` where ``raw`` is
            [B, action_dim] pre-tanh, ``env_action`` is [B, action_dim] in (-1, 1),
            ``log_prob`` is [B] log-density of the action, and ``value`` is [B] the
            critic estimate V(p_t, command).
        """
        latent = self._latent(proprio, command)
        dist = self._dist(latent)
        raw = dist.sample_raw()  # store pre-tanh u for a consistent PPO ratio
        return raw, TanhNormal.to_env(raw), dist.log_prob_from_raw(raw), self.value_head(latent).squeeze(-1)

    def evaluate_actions(self, proprio, command, raw):
        """Re-score stored stage-1 actions for the PPO update (gradients ON).

        The update-time counterpart to :meth:`act`: recomputes the log-prob, entropy,
        and value of previously collected raw samples so PPO can form the clipped
        surrogate loss and the value-function and entropy terms.

        Args:
            proprio: [B, proprio_dim] proprioceptive observation p_t from the batch.
            command: [B, command_dim] command from the batch.
            raw: [B, action_dim] pre-tanh sample stored during :meth:`act`.

        Returns:
            tuple ``(log_prob, entropy, value)`` with shapes [B], [B], [B]
            respectively -- the per-sample new log-density, entropy proxy, and value.
        """
        latent = self._latent(proprio, command)
        dist = self._dist(latent)
        return dist.log_prob_from_raw(raw), dist.entropy(), self.value_head(latent).squeeze(-1)

    @torch.no_grad()
    def value_only(self, proprio, command):
        """Evaluate just the critic V(p_t, command) -- used for GAE bootstrapping.

        Cheap value-only forward pass (e.g. to bootstrap the return at a rollout
        boundary in stage-1 PPO) that skips the action head.

        Args:
            proprio: [B, proprio_dim] proprioceptive observation p_t.
            command: [B, command_dim] command.

        Returns:
            [B] critic value estimate.
        """
        return self.value_head(self._latent(proprio, command)).squeeze(-1)

    # ----- stage 2: frozen, deterministic -----
    @torch.no_grad()
    def action(self, proprio, command):
        """Deterministic action tanh(mean) -- the frozen operator inside the HRL policy.

        The stage-2 entry point: with the LL frozen, the HRL policy queries the most-
        likely low-level action for the ML's command, turning the command into the
        actuator command a_t executed in the env. No sampling, no value head.

        Args:
            proprio: [B, proprio_dim] proprioceptive observation p_t.
            command: [B, command_dim] command from the Middle Layer (in (-1, 1)).

        Returns:
            [B, action_dim] deterministic low-level action in (-1, 1).
        """
        return torch.tanh(self.action_mean(self._latent(proprio, command)))

    def freeze(self):
        """Disable grads and switch to eval mode -- enter the stage-2 frozen regime.

        Implements the §5.2 freeze: sets ``requires_grad=False`` on every LL parameter
        and calls ``.eval()`` so the IPPO optimizer that trains UL+ML can never touch
        the operator. Returns ``self`` for fluent chaining.

        Returns:
            ``LowerLayer`` -- this same (now frozen) module.
        """
        for p in self.parameters():
            p.requires_grad_(False)  # IPPO will skip these via trainable_parameters()
        self.eval()                  # lock any train-mode behaviour (e.g. dropout/bn if added)
        return self

    def save(self, path: str):
        """Persist the LL weights and its stage-1 reward ``mode`` to disk.

        Serializes the state dict alongside ``mode`` so the operator can be reloaded
        with the correct paradigm tag for stage-2 cooperation training.

        Args:
            path: str -- destination file path for ``torch.save``.

        Returns:
            None.
        """
        torch.save({"state_dict": self.state_dict(), "mode": self.mode}, path)

    @classmethod
    def load_frozen(cls, path: str, *, proprio_dim: int, command_dim: int, action_dim: int,
                    hidden_size: int = 256, map_location="cpu"):
        """Reconstruct a pre-trained LL from disk and immediately freeze it.

        The bridge from stage 1 to stage 2: instantiates the operator with the given
        dimensions, restores the saved weights (and ``mode``), and freezes it so it is
        ready to plug into the HRL policy.

        Args:
            path: str -- checkpoint produced by :meth:`save`.
            proprio_dim: int -- proprioceptive obs width (must match training).
            command_dim: int -- command width (must match training).
            action_dim: int -- action width (must match training).
            hidden_size: int -- encoder hidden width (default 256, must match training).
            map_location: device spec passed to ``torch.load`` (default "cpu").

        Returns:
            ``LowerLayer`` -- a loaded, frozen operator.
        """
        payload = torch.load(path, map_location=map_location)
        ll = cls(proprio_dim, command_dim, action_dim, hidden_size,
                 mode=payload.get("mode", "position"))  # default to the paper's position paradigm
        ll.load_state_dict(payload["state_dict"])
        return ll.freeze()  # never train the operator again in stage 2
