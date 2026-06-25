# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""STAGE 1 runner (paper §5.2.1): pre-train the lower locomotion operator single-agent
with ``ppo_single``, then FREEZE (position mode) and save for stage 2.

Call chain:

    train_lower(cfg)
      └─ build single-agent LocomotionEnv (position|velocity)
      └─ run_lower(env_fn, cfg)
           LowerLayer  ──act──▶ env.step ──▶ collect rollout
           compute_gae_truncated  ──▶  ppo_update_lower  (clip-PPO on the LL)
           LowerLayer.freeze() + .save()   ──▶  results/lower/<mode>_operator.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dhrl.algorithms.ppo_single import normalize, ppo_update_lower
from dhrl.memories.rollout_buffer import compute_gae_truncated
from dhrl.networks.lower_layer import LowerLayer
from dhrl.utils.logging import StepLogger


def run_lower(env_fn, cfg: dict, *, device=None, results_dir="dhrl/results/lower", exp_name="lower"):
    """Run the full stage-1 PPO loop that pre-trains, then freezes and saves, the Lower Layer.

    This is the executable form of paper §5.2.1: collect single-agent locomotion rollouts,
    fit the LL actor+critic with truncation-aware GAE and clipped PPO, and on completion
    FREEZE the operator and serialize it to disk so stage 2 (``run_upper``) can load it as
    a fixed proprio+command -> action module under the trainable UL+ML hierarchy. The LL is
    trained under one paradigm (``cfg['paradigm']``): ``position`` (reward = reciprocal L2
    distance to a target, the operator the paper actually reuses) or ``velocity`` (reward =
    velocity dot target). Taking an ``env_fn`` factory rather than a live env lets tests and
    real runs share this code path unchanged.

    Args:
        env_fn (callable): zero-arg factory returning a single-agent locomotion env exposing
            ``proprio_dim`` / ``command_dim`` / ``action_dim``, plus ``reset`` -> (proprio, command)
            and ``step(action)`` -> (proprio, command, reward, done, info).
        cfg (dict): run config. Reads ``ppo`` (gamma, gae_lambda, lr, clip, epochs, minibatches),
            ``rollout_steps`` T, ``total_steps``, ``hidden_size`` H, ``paradigm``, ``seed``, ``log_every``.
        device: torch device; defaults to CUDA when available else CPU.
        results_dir (str): directory for the metrics CSV and the frozen operator checkpoint.
        exp_name (str): experiment tag used in the metrics-CSV filename.

    Returns:
        str: filesystem path of the saved frozen-operator checkpoint (``<paradigm>_operator.pt``),
        the artifact stage 2 consumes via ``LowerLayer.load_frozen``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = cfg.get("ppo", {})
    gamma, lam = ppo.get("gamma", 0.995), ppo.get("gae_lambda", 0.95)  # discount + GAE bias/variance trade-off
    T = cfg.get("rollout_steps", 256)                  # steps collected before each PPO update
    total = cfg.get("total_steps", 200_000)            # total env frames budget
    H = cfg.get("hidden_size", 256)                    # LL MLP width

    env = env_fn()
    # Build the LL sized to the env's spaces; ``mode`` selects which reward paradigm head/logic is active.
    lower = LowerLayer(env.proprio_dim, env.command_dim, env.action_dim, H,
                       mode=cfg.get("paradigm", "position")).to(device)
    opt = torch.optim.Adam(lower.parameters(), lr=ppo.get("lr", 5e-4), eps=1e-5)
    logger = StepLogger(Path(results_dir) / f"{exp_name}_seed_{cfg.get('seed', 0)}_metrics.csv",
                        log_every=cfg.get("log_every", 2000),
                        fields=["env_steps", "reward", "policy_loss", "value_loss", "entropy"])

    def to_t(x):
        """Convert array-like ``x`` to a float32 tensor on the training ``device``."""
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    proprio, command = env.reset()
    frames, ep_ret, recent = 0, 0.0, []                # global frame counter, running episode return, recent-return window
    while frames < total:
        # Per-iteration rollout buffers (Python lists, stacked into tensors at update time):
        # P=proprio, C=command, RAW=pre-squash action, LP=log-prob, VAL=value, REW=reward,
        # DONE=episode-end flag, BOOT=bootstrap value injected on time-limit truncations.
        P, C, RAW, LP, VAL, REW, DONE, BOOT = [], [], [], [], [], [], [], []
        for _ in range(T):
            # LL forward pass: sample an action and store the raw (pre-squash) sample + its log-prob + value.
            raw, env_a, lp, val = lower.act(to_t(proprio)[None], to_t(command)[None])
            npro, ncmd, r, d, info = env.step(env_a[0].cpu().numpy())  # advance the single env one step
            ep_ret += r
            boot = 0.0
            if d and info.get("truncated"):
                # Time-limit truncation (NOT a true terminal): bootstrap V at the cut state so the
                # return target stays consistent (truncation-aware GAE). The env exposes the
                # terminal proprio/command separately because ``npro``/``ncmd`` are already reset state.
                boot = float(lower.value_only(to_t(info["term_proprio"])[None],
                                              to_t(info["term_command"])[None])[0])
            P.append(proprio); C.append(command); RAW.append(raw[0].cpu().numpy())
            LP.append(float(lp[0])); VAL.append(float(val[0])); REW.append(float(r))
            DONE.append(float(d)); BOOT.append(boot)
            proprio, command = (npro, ncmd)            # roll observation forward
            if d:
                recent.append(ep_ret); ep_ret = 0.0    # close out the episode-return record
                proprio, command = env.reset()         # start a fresh episode
            frames += 1

        # Bootstrap value for the last (possibly mid-episode) state to terminate the GAE recursion.
        last_value = float(lower.value_only(to_t(proprio)[None], to_t(command)[None])[0])
        # Truncation-aware GAE: BOOT injects V at truncations so a time-limit cut does not zero the return.
        adv, ret = compute_gae_truncated(
            to_t(REW)[:, None], to_t(VAL)[:, None], to_t(DONE)[:, None],
            to_t(BOOT)[:, None], to_t([last_value]), gamma, lam)
        batch = {
            "proprio": to_t(np.asarray(P, np.float32)),     # [T, proprio_dim]
            "command": to_t(np.asarray(C, np.float32)),     # [T, command_dim]
            "raw_action": to_t(np.asarray(RAW, np.float32)),# [T, action_dim] pre-squash samples
            "old_log_prob": to_t(np.asarray(LP, np.float32)),# [T] behaviour-policy log-probs
            "advantage": normalize(adv.reshape(-1)),        # [T] standardized advantages
            "ret": ret.reshape(-1),                         # [T] value targets
        }
        losses = ppo_update_lower(lower, opt, batch, clip_param=ppo.get("clip", 0.2),
                                  ppo_epochs=ppo.get("epochs", 4),
                                  num_minibatches=ppo.get("minibatches", 4))
        if logger.should_log(frames):
            # Log mean of the last 50 episode returns (NaN until the first episode finishes) + PPO losses.
            logger.log(frames, {"reward": float(np.mean(recent[-50:])) if recent else float("nan"), **losses})

    lower.freeze()                                     # <-- the stage-1 deliverable: stop-grad the operator
    ckpt = Path(results_dir) / f"{cfg.get('paradigm', 'position')}_operator.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    lower.save(str(ckpt))                              # serialize for stage 2 to reload frozen
    env.close()
    return str(ckpt)


def train_lower(cfg: dict):
    """Build the single-agent locomotion env from the config, then run stage-1 PPO.

    Thin config->runner adapter and the ``stage == 'lower'`` entry point dispatched by
    ``run.main`` (§5.2.1). It instantiates the ``LocomotionEnv`` for the requested reward
    paradigm with a seeded RNG (reproducibility) and delegates the actual training loop to
    ``run_lower``; the result is the frozen locomotion operator consumed by stage 2.

    Args:
        cfg (dict): the loaded config; reads ``paradigm`` (position|velocity), ``seed``, and
            forwards everything else to ``run_lower`` (ppo hyperparams, step budgets, etc.).

    Returns:
        str: path to the saved frozen-operator checkpoint (forwarded from ``run_lower``).
    """
    from dhrl.envs.locomotion import LocomotionEnv

    def make():
        """Factory: a fresh single-agent ``LocomotionEnv`` for the configured paradigm + seed."""
        return LocomotionEnv(mode=cfg.get("paradigm", "position"),
                             rng=np.random.default_rng(cfg["seed"]))

    # ``exp_name`` embeds the paradigm so position/velocity runs write to distinct metric files.
    return run_lower(make, cfg, exp_name=f"lower_{cfg.get('paradigm', 'position')}")
