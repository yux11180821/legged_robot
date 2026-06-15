#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage 1: train the lower operator (算子) single-agent, then it gets FROZEN.

    python dhrl_habitat/lower_train.py --mode velocity --total-steps 500000 --seed 100
    python dhrl_habitat/lower_train.py --mode position --total-steps 500000 --seed 100

Outputs:
  checkpoints: data/checkpoints/<exp>/lower_<mode>_seed_<seed>.pt
  metrics CSV: results/<exp>/lower_<mode>_seed_<seed>_metrics.csv
               (a data point every --log-every env steps + inference time)
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

from habitat_commander.metrics import InferenceTimer, StepLogger  # noqa: E402
from habitat_commander.ppo import (  # noqa: E402
    PPOBatch,
    compute_gae_truncated,
    normalize_advantages,
    ppo_update,
)

from dhrl_habitat.habitat_io import make_dhrl_env  # noqa: E402
from dhrl_habitat.layers import LOWER_OBS_DIM, LowerOperator, save_lower_checkpoint  # noqa: E402
from dhrl_habitat.lower_env import LowerTrainAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 lower operator training.")
    p.add_argument("--config-name", default="benchmark/multi_agent/hssd_spot_spot.yaml")
    p.add_argument("--exp-name", default="dhrl_repro")
    p.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(Path.cwd())))
    p.add_argument("--mode", choices=["position", "velocity"], default="velocity")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--hidden-size", type=int, default=128)
    # paper hyperparameters: Adam 5e-4, clip 0.2, gamma 0.995
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=1e-3)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-interval", type=int, default=100_000)
    p.add_argument("--nominal-step-disp", type=float, default=10.0 / 120.0,
                   help="Per-step displacement at full command (preflight step 5 prints the measured value).")
    p.add_argument("--nominal-step-yaw", type=float, default=10.0 / 120.0,
                   help="Per-step yaw at full command (preflight step 5 prints the measured value).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    env = make_dhrl_env(
        config_name=args.config_name,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        action_keys=["agent_0_base_velocity"],  # single-agent stage: agent_1 stands still
    )
    adapter = LowerTrainAdapter(
        env, mode=args.mode, rng=rng,
        nominal_step_disp=args.nominal_step_disp,
        nominal_step_yaw=args.nominal_step_yaw,
    )

    lower = LowerOperator(obs_dim=LOWER_OBS_DIM, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(lower.parameters(), lr=args.lr, eps=1e-5)

    results_dir = Path(args.project_dir) / "results" / args.exp_name
    ckpt_dir = Path(args.project_dir) / "data" / "checkpoints" / args.exp_name
    logger = StepLogger(
        results_dir / f"lower_{args.mode}_seed_{args.seed}_metrics.csv",
        log_every=args.log_every,
    )
    timer = InferenceTimer(device=str(device))

    obs = adapter.reset()
    frames = 0
    next_ckpt = args.ckpt_interval
    recent_rewards: deque[float] = deque(maxlen=200)
    last_losses: dict[str, float] = {}

    print(f"stage1_start mode={args.mode} seed={args.seed} device={device}", flush=True)
    while frames < args.total_steps:
        obs_buf, act_buf, lp_buf, rew_buf, done_buf, val_buf, boot_buf = [], [], [], [], [], [], []

        for _ in range(args.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with timer.measure():  # 推理一步耗时
                raw, env_action, log_prob, value = lower.act(obs_t)

            next_obs, reward, done, _info = adapter.step(env_action[0].cpu().numpy())
            recent_rewards.append(reward)

            boot = 0.0
            if done:  # stage-1 episodes only end on the time limit -> truncation
                term_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                boot = float(lower.value_only(term_t).item())
                next_obs = adapter.reset()

            obs_buf.append(obs)
            act_buf.append(raw[0].cpu().numpy())
            lp_buf.append(float(log_prob[0].cpu()))
            rew_buf.append(reward)
            done_buf.append(float(done))
            val_buf.append(float(value[0].cpu()))
            boot_buf.append(boot)

            obs = next_obs
            frames += 1
            if logger.should_log(frames):
                logger.log(
                    frames,
                    {"reward": float(np.mean(recent_rewards)), **timer.stats(), **last_losses},
                )

        last_value = lower.value_only(
            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        ).cpu().numpy()

        adv, ret = compute_gae_truncated(
            np.asarray(rew_buf, np.float32)[:, None],
            np.asarray(val_buf, np.float32)[:, None],
            np.asarray(done_buf, np.float32)[:, None],
            np.asarray(boot_buf, np.float32)[:, None],
            last_value,
            args.gamma,
            args.gae_lambda,
        )
        batch = PPOBatch(
            obs=torch.as_tensor(np.asarray(obs_buf, np.float32), device=device),
            hidden=None,
            raw_actions=torch.as_tensor(np.asarray(act_buf, np.float32), device=device),
            old_log_probs=torch.as_tensor(np.asarray(lp_buf, np.float32), device=device),
            returns=torch.as_tensor(ret[:, 0], device=device),
            advantages=normalize_advantages(torch.as_tensor(adv[:, 0], device=device)),
        )
        losses = ppo_update(
            lower, optimizer, batch,
            clip_param=args.clip_param, entropy_coef=args.entropy_coef,
            ppo_epochs=args.ppo_epochs, num_minibatches=args.num_minibatches,
        )
        last_losses = {k: losses[k] for k in ("policy_loss", "value_loss", "entropy")}

        if frames >= next_ckpt:
            save_lower_checkpoint(
                str(ckpt_dir / f"lower_{args.mode}_seed_{args.seed}_{frames:09d}.pt"),
                lower, mode=args.mode, meta={"frames": frames, "seed": args.seed},
            )
            next_ckpt += args.ckpt_interval

    final = ckpt_dir / f"lower_{args.mode}_seed_{args.seed}.pt"
    save_lower_checkpoint(str(final), lower, mode=args.mode,
                          meta={"frames": frames, "seed": args.seed})
    print(f"stage1_done frames={frames} checkpoint={final}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
