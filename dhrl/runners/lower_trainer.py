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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = cfg.get("ppo", {})
    gamma, lam = ppo.get("gamma", 0.995), ppo.get("gae_lambda", 0.95)
    T = cfg.get("rollout_steps", 256)
    total = cfg.get("total_steps", 200_000)
    H = cfg.get("hidden_size", 256)

    env = env_fn()
    lower = LowerLayer(env.proprio_dim, env.command_dim, env.action_dim, H,
                       mode=cfg.get("paradigm", "position")).to(device)
    opt = torch.optim.Adam(lower.parameters(), lr=ppo.get("lr", 5e-4), eps=1e-5)
    logger = StepLogger(Path(results_dir) / f"{exp_name}_seed_{cfg.get('seed', 0)}_metrics.csv",
                        log_every=cfg.get("log_every", 2000),
                        fields=["env_steps", "reward", "policy_loss", "value_loss", "entropy"])

    def to_t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    proprio, command = env.reset()
    frames, ep_ret, recent = 0, 0.0, []
    while frames < total:
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
            proprio, command = (npro, ncmd)
            if d:
                recent.append(ep_ret); ep_ret = 0.0
                proprio, command = env.reset()
            frames += 1

        last_value = float(lower.value_only(to_t(proprio)[None], to_t(command)[None])[0])
        adv, ret = compute_gae_truncated(
            to_t(REW)[:, None], to_t(VAL)[:, None], to_t(DONE)[:, None],
            to_t(BOOT)[:, None], to_t([last_value]), gamma, lam)
        batch = {
            "proprio": to_t(np.asarray(P, np.float32)),
            "command": to_t(np.asarray(C, np.float32)),
            "raw_action": to_t(np.asarray(RAW, np.float32)),
            "old_log_prob": to_t(np.asarray(LP, np.float32)),
            "advantage": normalize(adv.reshape(-1)),
            "ret": ret.reshape(-1),
        }
        losses = ppo_update_lower(lower, opt, batch, clip_param=ppo.get("clip", 0.2),
                                  ppo_epochs=ppo.get("epochs", 4),
                                  num_minibatches=ppo.get("minibatches", 4))
        if logger.should_log(frames):
            logger.log(frames, {"reward": float(np.mean(recent[-50:])) if recent else float("nan"), **losses})

    lower.freeze()
    ckpt = Path(results_dir) / f"{cfg.get('paradigm', 'position')}_operator.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    lower.save(str(ckpt))
    env.close()
    return str(ckpt)


def train_lower(cfg: dict):
    """Build the single-agent locomotion env from the config, then run stage-1 PPO."""
    from dhrl.envs.locomotion import LocomotionEnv

    def make():
        return LocomotionEnv(mode=cfg.get("paradigm", "position"),
                             rng=np.random.default_rng(cfg["seed"]))

    return run_lower(make, cfg, exp_name=f"lower_{cfg.get('paradigm', 'position')}")
