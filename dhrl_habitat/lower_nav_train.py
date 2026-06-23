#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""LOWER LAYER: single-agent PPO trains ONE Spot to navigate to a goal point.

Corrected plan (2026-06-17): all training in Habitat, NO distributed.  A single
PPO policy outputs one Spot's (agent_0) holonomic base velocity for point-goal
navigation; the other Spot (agent_1) follows a SCRIPTED NavMesh walk-to-goal
rule.  Train one Spot at a time, then FREEZE -> the upper MAPPO layer commands it.

Reuses the bug-fixed single-agent PPO core (TanhNormal action in (-1,1) +
truncation-aware GAE + LR anneal) and the metrics harness.  Several Habitat envs
are stepped sequentially so each PPO update sees a [T, N] batch.

    python dhrl_habitat/lower_nav_train.py --num-envs 4 --total-steps 500000
    python dhrl_habitat/lower_nav_train.py --smoke      # tiny CPU/GPU smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from habitat_commander.commander_policy import CommanderPolicy  # noqa: E402
from habitat_commander.metrics import InferenceTimer, StepLogger  # noqa: E402
from habitat_commander.ppo import (  # noqa: E402
    PPOBatch, compute_gae_truncated, normalize_advantages, ppo_update,
)

from dhrl_habitat.habitat_io import make_dhrl_env  # noqa: E402
from dhrl_habitat.lower_nav_env import ACT_DIM, OBS_DIM, LowerNavAdapter  # noqa: E402

ACTION_KEYS = ["agent_0_base_velocity", "agent_1_base_velocity"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lower-layer single-agent PPO point-goal nav (Habitat).")
    p.add_argument("--config", default="benchmark/multi_agent/replica_cad_spot_spot.yaml")
    p.add_argument("--exp-name", default="lower_nav")
    p.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(Path.cwd())))
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=1e-3)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--ckpt-interval", type=int, default=100_000)
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the whole loop")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_envs = 1
        args.total_steps = 200
        args.rollout_steps = 16
        args.max_episode_steps = 60
        args.log_every = 32
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.num_envs

    envs = [
        make_dhrl_env(
            config_name=args.config, seed=args.seed + i,
            max_episode_steps=args.max_episode_steps,
            action_keys=ACTION_KEYS, enable_lateral=True,
        )
        for i in range(N)
    ]
    adapters = [
        LowerNavAdapter(envs[i], rng=np.random.default_rng(args.seed * 1000 + i))
        for i in range(N)
    ]

    policy = CommanderPolicy(obs_dim=OBS_DIM, action_dim=ACT_DIM,
                             hidden_size=args.hidden_size, use_memory=False).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    results_dir = Path(args.project_dir) / "results" / args.exp_name
    ckpt_dir = Path(args.project_dir) / "data" / "checkpoints" / args.exp_name
    logger = StepLogger(
        results_dir / f"lower_nav_seed_{args.seed}_metrics.csv", log_every=args.log_every,
        fields=["env_steps", "reward", "success_rate", "ep_len",
                "infer_ms_mean", "infer_ms_median", "infer_ms_p95", "infer_ms_last",
                "policy_loss", "value_loss", "entropy"],
    )
    timer = InferenceTimer(device=str(device))

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    obs = np.stack([ad.reset() for ad in adapters]).astype(np.float32)  # [N, OBS_DIM]
    frames, next_ckpt = 0, args.ckpt_interval
    ep_ret = np.zeros(N, dtype=np.float32)
    ep_len = np.zeros(N, dtype=np.int64)
    recent_ret: deque[float] = deque(maxlen=100)
    recent_len: deque[float] = deque(maxlen=100)
    recent_succ: deque[float] = deque(maxlen=100)
    last_losses: dict[str, float] = {}

    print(f"lower_nav start envs={N} obs={OBS_DIM} act={ACT_DIM} device={device}", flush=True)
    while frames < args.total_steps:
        b_obs, b_raw, b_lp, b_rew, b_done, b_boot, b_val = [], [], [], [], [], [], []
        for _ in range(args.rollout_steps):
            with timer.measure():
                raw, env_a, lp, val, _ = policy.act(to_t(obs), None)
            actions = env_a.cpu().numpy().reshape(N, ACT_DIM)

            next_obs = np.zeros_like(obs)
            rew = np.zeros(N, dtype=np.float32)
            done = np.zeros(N, dtype=np.float32)
            boots = np.zeros(N, dtype=np.float32)
            term_obs_list, trunc_idx = [], []
            for i, ad in enumerate(adapters):
                no, r, d, info = ad.step(actions[i])
                next_obs[i] = no
                rew[i] = r
                done[i] = float(d)
                ep_ret[i] += r
                ep_len[i] += 1
                if d:
                    recent_ret.append(float(ep_ret[i]))
                    recent_len.append(float(ep_len[i]))
                    recent_succ.append(1.0 if info["success"] else 0.0)
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                if info["truncated"]:
                    trunc_idx.append(i)
                    term_obs_list.append(info["term_obs"])
            if trunc_idx:
                bv = policy.value_only(to_t(np.stack(term_obs_list)), None).cpu().numpy()
                for k, i in enumerate(trunc_idx):
                    boots[i] = bv[k]

            b_obs.append(obs.copy())
            b_raw.append(raw.cpu().numpy())
            b_lp.append(lp.cpu().numpy())
            b_rew.append(rew.copy())
            b_done.append(done.copy())
            b_boot.append(boots.copy())
            b_val.append(val.cpu().numpy())

            obs = next_obs
            frames += N
            if logger.should_log(frames):
                logger.log(frames, {
                    "reward": float(np.mean(recent_ret)) if recent_ret else float("nan"),
                    "success_rate": float(np.mean(recent_succ)) if recent_succ else float("nan"),
                    "ep_len": float(np.mean(recent_len)) if recent_len else float("nan"),
                    **timer.stats(), **last_losses})

        last_value = policy.value_only(to_t(obs), None).cpu().numpy()
        adv, ret = compute_gae_truncated(
            np.asarray(b_rew, np.float32), np.asarray(b_val, np.float32),
            np.asarray(b_done, np.float32), np.asarray(b_boot, np.float32),
            last_value, args.gamma, args.gae_lambda)

        if not args.no_anneal_lr:
            lr_now = args.lr * max(0.0, 1.0 - frames / args.total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
        batch = PPOBatch(
            obs=to_t(np.asarray(b_obs, np.float32)).flatten(0, 1),
            hidden=None,
            raw_actions=to_t(np.asarray(b_raw, np.float32)).flatten(0, 1),
            old_log_probs=to_t(np.asarray(b_lp, np.float32)).flatten(0, 1),
            returns=to_t(ret.reshape(-1)),
            advantages=normalize_advantages(to_t(adv.reshape(-1))))
        losses = ppo_update(policy, optimizer, batch,
                            clip_param=args.clip_param, entropy_coef=args.entropy_coef,
                            ppo_epochs=args.ppo_epochs, num_minibatches=args.num_minibatches)
        last_losses = {k: losses[k] for k in ("policy_loss", "value_loss", "entropy")}

        if frames >= next_ckpt:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"policy": policy.state_dict(), "frames": frames, "seed": args.seed,
                        "obs_dim": OBS_DIM, "action_dim": ACT_DIM, "hidden_size": args.hidden_size},
                       ckpt_dir / f"lower_nav_seed_{args.seed}_{frames:09d}.pt")
            next_ckpt += args.ckpt_interval

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "frames": frames, "seed": args.seed,
                "obs_dim": OBS_DIM, "action_dim": ACT_DIM, "hidden_size": args.hidden_size},
               ckpt_dir / f"lower_nav_seed_{args.seed}_final.pt")
    for e in envs:
        try:
            e.close()
        except Exception:
            pass
    print(f"lower_nav done frames={frames}", flush=True)


if __name__ == "__main__":
    main()
