#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""CENTRALIZED Go2 walker baseline (one policy controls all 12 joints).

Diagnostic: if a centralized policy learns to walk in this env/reward, then the
env+reward are sound and the per-leg DECENTRALIZED case is the (hard) research
contribution.  If centralized ALSO fails, the env/reward recipe needs more work.

Single-agent PPO over the full robot: obs = the env's 52-dim global state,
action = 12 joint position targets.  Reuses CommanderPolicy + the bug-fixed
single-agent PPO (truncation-aware GAE) + LR annealing + metrics.

    python mujoco_leg_marl/train_central.py --num-envs 32 --total-steps 3000000
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

from mujoco_leg_marl.go2_leg_env import (  # noqa: E402
    LEG_NAMES, NUM_JOINTS, STATE_DIM, Go2LegMARLCfg, Go2LegMARLEnv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centralized Go2 walker baseline (MuJoCo).")
    p.add_argument("--model-path", default="mujoco_menagerie/unitree_go2/scene.xml")
    p.add_argument("--exp-name", default="go2_leg_central")
    p.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(Path.cwd())))
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=5e-4)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--ckpt-interval", type=int, default=500_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, A = args.num_envs, len(LEG_NAMES)

    env = Go2LegMARLEnv(Go2LegMARLCfg(model_path=args.model_path, num_envs=N))
    env.rng = np.random.default_rng(args.seed)

    # centralized policy: full 52-dim state -> 12 joint actions
    policy = CommanderPolicy(obs_dim=STATE_DIM, action_dim=NUM_JOINTS,
                             hidden_size=args.hidden_size, use_memory=False).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    results_dir = Path(args.project_dir) / "results" / args.exp_name
    ckpt_dir = Path(args.project_dir) / "data" / "checkpoints" / args.exp_name
    logger = StepLogger(results_dir / f"central_seed_{args.seed}_metrics.csv", log_every=args.log_every,
                        fields=["env_steps", "reward", "ep_len",
                                "infer_ms_mean", "infer_ms_median", "infer_ms_p95", "infer_ms_last",
                                "policy_loss", "value_loss", "entropy"])
    timer = InferenceTimer(device=str(device))

    _, state = env.reset()
    frames, next_ckpt = 0, args.ckpt_interval
    ep_ret = np.zeros(N, dtype=np.float32)
    ep_len = np.zeros(N, dtype=np.int64)
    recent_ret: deque[float] = deque(maxlen=100)
    recent_len: deque[float] = deque(maxlen=100)
    last_losses: dict[str, float] = {}

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    print(f"go2_central start envs={N} obs={STATE_DIM} act={NUM_JOINTS} device={device}", flush=True)
    while frames < args.total_steps:
        b_obs, b_raw, b_lp, b_rew, b_done, b_boot, b_val = [], [], [], [], [], [], []
        for _ in range(args.rollout_steps):
            with timer.measure():
                raw, env_a, lp, val, _ = policy.act(to_t(state), None)
            actions = env_a.cpu().numpy().reshape(N, A, 3)
            _, nxt_state, reward, done, info = env.step(actions)
            team_rew = reward.mean(axis=1)
            ep_ret += team_rew; ep_len += 1

            boots = np.zeros(N, dtype=np.float32)
            trunc = np.nonzero(info["truncated"])[0]
            if len(trunc):
                boots[trunc] = policy.value_only(to_t(info["term_state"][trunc]), None).cpu().numpy()
            for e in np.nonzero(done)[0]:
                recent_ret.append(float(ep_ret[e])); recent_len.append(float(ep_len[e]))
                ep_ret[e] = 0.0; ep_len[e] = 0

            b_obs.append(state.copy()); b_raw.append(raw.cpu().numpy()); b_lp.append(lp.cpu().numpy())
            b_rew.append(team_rew.copy()); b_done.append(done.astype(np.float32)); b_boot.append(boots)
            b_val.append(val.cpu().numpy())
            state = nxt_state
            frames += N
            if logger.should_log(frames):
                logger.log(frames, {
                    "reward": float(np.mean(recent_ret)) if recent_ret else float("nan"),
                    "ep_len": float(np.mean(recent_len)) if recent_len else float("nan"),
                    **timer.stats(), **last_losses})

        last_value = policy.value_only(to_t(state), None).cpu().numpy()
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
            torch.save({"policy": policy.state_dict(), "frames": frames, "seed": args.seed},
                       ckpt_dir / f"central_seed_{args.seed}_{frames:09d}.pt")
            next_ckpt += args.ckpt_interval

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "frames": frames, "seed": args.seed},
               ckpt_dir / f"central_seed_{args.seed}_final.pt")
    print(f"go2_central done frames={frames}", flush=True)


if __name__ == "__main__":
    main()
