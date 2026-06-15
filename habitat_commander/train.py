#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Train the upper-layer Commander in Habitat: perception -> (vx, vy, wz).

Pipeline (matches "VLM offline distillation -> PPO end-to-end"):
  * optional --bc-warmstart: collect (obs, teacher_command) with a VLMTeacher
    (default: dependency-free OracleGoalTeacher) and behavior-clone the policy;
  * PPO fine-tune with truncation-aware GAE and a tanh-squashed action head.

The action drives Habitat's kinematic base-velocity controller -- the "ideal
frozen lower layer" during Commander training.  At deployment the same (vx,vy,wz)
drives the IsaacLab per-leg MARL lower layer (see ../isaaclab_lowlevel).

Run on AutoDL (Habitat env + data):
    python habitat_commander/train.py --config-name benchmark/multi_agent/hssd_spot_human_social_nav.yaml \
        --bc-warmstart --total-steps 2000000
(Adapt --config-name / --depth-key / --goal-key / --action-key to your env; see README.)
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from commander_policy import CommanderPolicy
from habitat_env import CommanderObs, CommanderReward, collision_from_info, make_commander_env
from metrics import InferenceTimer, StepLogger
from ppo import PPOBatch, compute_gae_truncated, normalize_advantages, ppo_update
from vlm_distill import ManualCommandPolicy, behavior_clone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Habitat Commander (upper layer).")
    p.add_argument("--config-name", default="benchmark/multi_agent/hssd_spot_human_social_nav.yaml")
    p.add_argument("--exp-name", default="commander_spot_nav")
    p.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(Path.cwd())))
    p.add_argument("--action-key", default="agent_0_base_velocity",
                   help="Holonomic base_velocity action key (enable_lateral_move=True).")
    p.add_argument("--depth-key", default="agent_0_spot_head_stereo_depth_sensor")
    p.add_argument("--goal-key", default="agent_0_goal_to_agent_gps_compass",
                   help="A (rho, phi) compass sensor your episodes expose.")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--no-memory", action="store_true", help="Disable the GRU memory.")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--value-loss-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=20, help="Record a data point every N env-steps (meeting requirement).")
    # VLM distillation
    p.add_argument("--bc-warmstart", action="store_true", help="Behavior-clone a VLM teacher first.")
    p.add_argument("--bc-steps", type=int, default=20_000, help="Env steps collected for BC.")
    p.add_argument("--bc-epochs", type=int, default=50)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_target(teacher: ManualCommandPolicy, rho: float, phi: float) -> np.ndarray:
    return teacher({"pointgoal_with_gps_compass": np.array([[rho, phi]], dtype=np.float32)})[0]


def collect_bc_dataset(args, envs, obs_builder, teacher, device):
    """Roll out with the teacher's command and record (flat_obs, teacher_command)."""
    flat_obs_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    obs_list = [env.reset() for env in envs]
    steps = 0
    while steps < args.bc_steps:
        for env_idx, env in enumerate(envs):
            obs = obs_list[env_idx]
            rho, phi = obs_builder.goal_rho_phi(obs)
            target = teacher_target(teacher, rho, phi)  # (vx, vy, wz) in [-1,1]
            flat_obs_list.append(obs_builder.transform(obs))
            target_list.append(target)
            next_obs, _, done, _ = env.step(target.astype(np.float32))
            obs_list[env_idx] = env.reset() if done else next_obs
            steps += 1
    flat = torch.as_tensor(np.asarray(flat_obs_list, dtype=np.float32), device=device)
    tgt = torch.as_tensor(np.asarray(target_list, dtype=np.float32), device=device)
    return flat, tgt


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = [make_commander_env(
        config_name=args.config_name, seed=args.seed + i,
        max_episode_steps=args.max_episode_steps, action_key=args.action_key,
        project_dir=args.project_dir,
    ) for i in range(max(1, args.num_envs))]

    obs_builder = CommanderObs(args.depth_key, args.goal_key, image_size=args.image_size)
    rewarders = [CommanderReward() for _ in envs]

    # Probe obs dim.
    obs_list = [env.reset() for env in envs]
    obs_features = np.stack([obs_builder.transform(o) for o in obs_list])
    for i, o in enumerate(obs_list):
        rewarders[i].reset(obs_builder.goal_rho_phi(o)[0])
    obs_dim = int(obs_features.shape[1])

    policy = CommanderPolicy(
        obs_dim=obs_dim, action_dim=3, hidden_size=args.hidden_size, use_memory=not args.no_memory
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    # ---- metrics harness (meeting requirements: 20-step points + single-step inference time) ----
    timer = InferenceTimer(device=str(device))
    csv_path = Path(args.project_dir) / "results" / args.exp_name / f"seed_{args.seed}_metrics.csv"
    logger = StepLogger(csv_path, log_every=args.log_every)
    last_losses: dict[str, float] = {}

    # ---- manual-command warm start (BC); VLM route deprioritized per meeting ----
    if args.bc_warmstart:
        teacher = ManualCommandPolicy()  # hand-crafted 人工指令 (swap a real VLM here only if revisited)
        print(f"[bc] collecting {args.bc_steps} steps from teacher...")
        bc_obs, bc_tgt = collect_bc_dataset(args, envs, obs_builder, teacher, device)
        info = behavior_clone(policy, bc_obs, bc_tgt, epochs=args.bc_epochs, lr=args.lr, device=device)
        print(f"[bc] done: {info}")
        # fresh rollout state after BC
        obs_list = [env.reset() for env in envs]
        obs_features = np.stack([obs_builder.transform(o) for o in obs_list])
        for i, o in enumerate(obs_list):
            rewarders[i].reset(obs_builder.goal_rho_phi(o)[0])

    hidden = policy.initial_hidden(len(envs), device)
    frames, update_idx = 0, 0
    recent_returns: deque[float] = deque(maxlen=50)
    running_return = np.zeros(len(envs), dtype=np.float32)

    while frames < args.total_steps:
        obs_buf, hid_buf, act_buf, lp_buf = [], ([] if policy.use_memory else None), [], []
        rew_buf, done_buf, val_buf, boot_buf = [], [], [], []

        for _ in range(args.rollout_steps):
            obs_t = torch.as_tensor(obs_features, dtype=torch.float32, device=device)
            if hid_buf is not None:
                hid_buf.append(hidden.detach().cpu().numpy())
            with timer.measure():  # single-step inference time (推理一步耗时)
                raw, env_action, log_prob, value, next_hidden = policy.act(obs_t, hidden)
            env_action_np = env_action.cpu().numpy().astype(np.float32)

            rewards = np.zeros(len(envs), dtype=np.float32)
            dones = np.zeros(len(envs), dtype=np.float32)
            bootstraps = np.zeros(len(envs), dtype=np.float32)
            next_features: list[np.ndarray] = []
            for env_idx, env in enumerate(envs):
                next_obs, _, done, info = env.step(env_action_np[env_idx])
                rho, _ = obs_builder.goal_rho_phi(next_obs)
                reward, reached = rewarders[env_idx].step(rho, collision_from_info(info))
                rewards[env_idx] = reward
                running_return[env_idx] += reward
                # Truncation-aware: success = true terminal (no bootstrap); any other
                # done (time limit / collision-terminate) = truncation (bootstrap V).
                truncated = bool(done) and not reached
                if done:
                    dones[env_idx] = 1.0
                    if truncated:
                        term_feat = obs_builder.transform(next_obs)
                        with torch.no_grad():
                            tv = policy.value_only(
                                torch.as_tensor(term_feat, dtype=torch.float32, device=device).unsqueeze(0),
                                None if hidden is None else hidden[env_idx : env_idx + 1],
                            )
                        bootstraps[env_idx] = float(tv.item())
                    recent_returns.append(float(running_return[env_idx]))
                    running_return[env_idx] = 0.0
                    next_obs = env.reset()
                    rewarders[env_idx].reset(obs_builder.goal_rho_phi(next_obs)[0])
                    if next_hidden is not None:
                        next_hidden[env_idx].zero_()
                next_features.append(obs_builder.transform(next_obs))

            obs_buf.append(obs_features.copy())
            act_buf.append(raw.cpu().numpy())
            lp_buf.append(log_prob.cpu().numpy())
            rew_buf.append(rewards)
            done_buf.append(dones)
            val_buf.append(value.cpu().numpy())
            boot_buf.append(bootstraps)

            obs_features = np.stack(next_features)
            hidden = None if next_hidden is None else next_hidden.detach()
            frames += len(envs)

            # Record a data point every --log-every env-steps (default 20).
            if logger.should_log(frames):
                mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
                logger.log(frames, {"reward": mean_ret, **timer.stats(), **last_losses})

        last_value = policy.value_only(
            torch.as_tensor(obs_features, dtype=torch.float32, device=device), hidden
        ).cpu().numpy()

        advantages, returns = compute_gae_truncated(
            np.asarray(rew_buf, np.float32), np.asarray(val_buf, np.float32),
            np.asarray(done_buf, np.float32), np.asarray(boot_buf, np.float32),
            last_value, args.gamma, args.gae_lambda,
        )

        obs_b = torch.as_tensor(np.asarray(obs_buf, np.float32), device=device).flatten(0, 1)
        act_b = torch.as_tensor(np.asarray(act_buf, np.float32), device=device).flatten(0, 1)
        lp_b = torch.as_tensor(np.asarray(lp_buf, np.float32), device=device).flatten(0, 1)
        ret_b = torch.as_tensor(returns, device=device).flatten(0, 1)
        adv_b = normalize_advantages(torch.as_tensor(advantages, device=device).flatten(0, 1))
        hid_b = None if hid_buf is None else torch.as_tensor(np.asarray(hid_buf, np.float32), device=device).flatten(0, 1)

        losses = ppo_update(
            policy, optimizer, PPOBatch(obs_b, hid_b, act_b, lp_b, ret_b, adv_b),
            clip_param=args.clip_param, value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs, num_minibatches=args.num_minibatches,
        )
        update_idx += 1
        last_losses = {k: losses[k] for k in ("policy_loss", "value_loss", "entropy") if k in losses}

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
