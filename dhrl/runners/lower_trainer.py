# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""STAGE 1 training loop (paper §5.2.1): pre-train the lower locomotion operator
single-agent with PPO, then FREEZE (position mode) and save for stage 2.

One ``train_lower(args)`` function, read top-to-bottom: a single-agent rollout through the
locomotion task -> truncation-aware GAE -> clipped-PPO update of the LowerLayer -> freeze + save.
"""
import os

import numpy as np
import torch

from algorithms.ppo_single import normalize, ppo_update_lower
from envs.locomotion import LocomotionEnv
from memories.rollout_buffer import compute_gae_truncated
from networks.lower_layer import LowerLayer
from utils.logging import StepLogger


def train_lower(args):
    """Pre-train + freeze the lower operator; returns the saved checkpoint path.

    Args:
        args: flat config (paradigm 'position'|'velocity', hidden_dim, lr, gamma,
            gae_lambda, clip, ppo_epochs, num_minibatches, rollout_steps, total_steps,
            run_dir, seed, log_every, device).
    """
    device = torch.device(getattr(args, "device", "cpu"))
    paradigm = getattr(args, "paradigm", "position")
    T = getattr(args, "rollout_steps", 256)
    gamma, lam = getattr(args, "gamma", 0.995), getattr(args, "gae_lambda", 0.95)

    env = LocomotionEnv(mode=paradigm, rng=np.random.default_rng(args.seed))
    lower = LowerLayer(env.proprio_dim, env.command_dim, env.action_dim,
                       getattr(args, "hidden_dim", 256), mode=paradigm).to(device)
    opt = torch.optim.Adam(lower.parameters(), lr=getattr(args, "lr", 5e-4), eps=1e-5)

    run_dir = getattr(args, "run_dir", "results")
    os.makedirs(run_dir, exist_ok=True)
    logger = StepLogger(os.path.join(run_dir, f"lower_{paradigm}_seed{args.seed}_metrics.csv"),
                        log_every=getattr(args, "log_every", 2000),
                        fields=["env_steps", "reward", "policy_loss", "value_loss", "entropy"])

    def to_t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    proprio, command = env.reset()
    steps, ep_ret, recent = 0, 0.0, []
    while steps < args.total_steps:
        P, C, RAW, LP, VAL, REW, DONE, BOOT = [], [], [], [], [], [], [], []
        for _ in range(T):
            raw, env_a, lp, val = lower.act(to_t(proprio)[None], to_t(command)[None])
            npro, ncmd, r, d, info = env.step(env_a[0].cpu().numpy())
            ep_ret += r
            boot = 0.0
            if d and info.get("truncated"):
                boot = float(lower.value_only(to_t(info["term_proprio"])[None],
                                              to_t(info["term_command"])[None])[0])
            P.append(proprio); C.append(command); RAW.append(raw[0].cpu().numpy())
            LP.append(float(lp[0])); VAL.append(float(val[0])); REW.append(float(r))
            DONE.append(float(d)); BOOT.append(boot)
            proprio, command = npro, ncmd
            if d:
                recent.append(ep_ret); ep_ret = 0.0
                proprio, command = env.reset()
            steps += 1

        last_v = float(lower.value_only(to_t(proprio)[None], to_t(command)[None])[0])
        # GAE is computed in torch (shared with IPPO); arrange the single-agent rollout as [T, 1].
        adv, ret = compute_gae_truncated(
            to_t(np.asarray(REW, np.float32)[:, None]), to_t(np.asarray(VAL, np.float32)[:, None]),
            to_t(np.asarray(DONE, np.float32)[:, None]), to_t(np.asarray(BOOT, np.float32)[:, None]),
            to_t([last_v]), gamma, lam)
        batch = {
            "proprio": to_t(np.asarray(P, np.float32)),
            "command": to_t(np.asarray(C, np.float32)),
            "raw_action": to_t(np.asarray(RAW, np.float32)),
            "old_log_prob": to_t(np.asarray(LP, np.float32)),
            "advantage": normalize(adv.reshape(-1)),
            "ret": ret.reshape(-1),
        }
        losses = ppo_update_lower(lower, opt, batch, clip_param=getattr(args, "clip", 0.2),
                                  ppo_epochs=getattr(args, "ppo_epochs", 4),
                                  num_minibatches=getattr(args, "num_minibatches", 4))
        if logger.should_log(steps):
            logger.log(steps, {"reward": float(np.mean(recent[-50:])) if recent else float("nan"), **losses})

    lower.freeze()
    ckpt = os.path.join(run_dir, f"{paradigm}_operator.pt")
    lower.save(ckpt)
    env.close()
    return ckpt
