#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage 2: CTDE MARL training of the Upper+Middle layers over the FROZEN operator.

Algorithms (paper Fig.5 / Table 1):
    dhrl          UL + ML(GRU) + frozen LL     (the proposed method)
    no_memory     UL + ML(no GRU) + frozen LL  (No Spatiotemporal Memory)
    no_hierarchy  flat MLP -> base velocity    (No Hierarchy; --lower-ckpt unused)

    python dhrl_habitat/upper_train.py --algorithm dhrl --seed 100 \
        --lower-ckpt data/checkpoints/dhrl_repro/lower_velocity_seed_100.pt

Execution is decentralized (each agent's actor sees only its own partial obs);
training uses one centralized critic over the joint obs (CTDE).  Homogeneous
agents share one actor (paper Eq.1).  Metrics: a CSV data point every
--log-every env steps incl. the single-step inference time of the FULL
hierarchy (UL+ML+LL forward).
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
from habitat_commander.ppo import normalize_advantages  # noqa: E402

from dhrl_habitat.habitat_io import make_dhrl_env  # noqa: E402
from dhrl_habitat.layers import (  # noqa: E402
    ACTION_DIM,
    ALGORITHMS,
    EXO_OBS_DIM,
    CentralCritic,
    build_actor,
    build_lower_obs,
    load_frozen_lower,
)
from dhrl_habitat.marl_ppo import (  # noqa: E402
    MARLBatch,
    broadcast_team_advantage,
    compute_gae_truncated,
    marl_ppo_update,
)
from dhrl_habitat.multi_env import MultiRobotNavAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 CTDE MARL training.")
    p.add_argument("--config-name", default="benchmark/multi_agent/hssd_spot_spot.yaml")
    p.add_argument("--exp-name", default="dhrl_repro")
    p.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(Path.cwd())))
    p.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="dhrl")
    p.add_argument("--lower-ckpt", default=None,
                   help="Frozen stage-1 operator (required for dhrl / no_memory).")
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--goal-mode", choices=["swap", "shared", "random"], default="swap")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--hidden-size", type=int, default=256)
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ALGORITHMS[args.algorithm]

    # ---- frozen lower operator (hierarchical algorithms only) ----
    lower, lower_mode = None, None
    if spec.hierarchical:
        if not args.lower_ckpt:
            raise SystemExit(f"--lower-ckpt is required for algorithm={args.algorithm}")
        lower, lower_mode = load_frozen_lower(args.lower_ckpt, device)

    # ---- envs + adapters ----
    action_keys = [f"agent_{i}_base_velocity" for i in range(args.n_agents)]
    envs, adapters = [], []
    for i in range(max(1, args.num_envs)):
        env = make_dhrl_env(
            config_name=args.config_name, seed=args.seed + i,
            max_episode_steps=args.max_episode_steps, action_keys=action_keys,
        )
        envs.append(env)
        adapters.append(MultiRobotNavAdapter(
            env, n_agents=args.n_agents, rng=np.random.default_rng(args.seed + i),
            goal_mode=args.goal_mode,
        ))
    n_envs, N = len(envs), args.n_agents
    joint_dim = N * EXO_OBS_DIM

    # ---- networks ----
    actor, _ = build_actor(args.algorithm, exo_dim=EXO_OBS_DIM, hidden_size=args.hidden_size)
    actor.to(device)
    critic = CentralCritic(joint_dim, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5
    )

    # ---- metrics ----
    results_dir = Path(args.project_dir) / "results" / args.exp_name
    ckpt_dir = Path(args.project_dir) / "data" / "checkpoints" / args.exp_name
    logger = StepLogger(
        results_dir / f"{args.algorithm}_seed_{args.seed}_metrics.csv",
        log_every=args.log_every,
        fields=["env_steps", "reward", "success_rate", "arrived",
                "infer_ms_mean", "infer_ms_median", "infer_ms_p95", "infer_ms_last",
                "policy_loss", "value_loss", "entropy"],
    )
    timer = InferenceTimer(device=str(device))

    # ---- state ----
    exo = np.stack([ad.reset() for ad in adapters])            # [E, N, exo]
    last_act = np.zeros((n_envs, N, ACTION_DIM), dtype=np.float32)
    hidden = actor.initial_hidden(n_envs * N, device)          # [E*N, H] | None
    frames = 0
    next_ckpt = args.ckpt_interval
    ep_returns = np.zeros(n_envs, dtype=np.float32)
    recent_returns: deque[float] = deque(maxlen=50)
    recent_success: deque[float] = deque(maxlen=50)
    last_losses: dict[str, float] = {}
    last_arrived = 0.0

    def joint_of(exo_arr: np.ndarray) -> np.ndarray:           # [E, N, exo] -> [E, joint]
        return exo_arr.reshape(exo_arr.shape[0], -1)

    def full_inference(exo_t: torch.Tensor, hid):
        """Batched decentralized act: commands + base velocities.  exo_t: [E*N, exo]."""
        raw, cmd, logp, nxt_hidden = actor.act(exo_t, hid)
        if spec.hierarchical:
            lower_obs = build_lower_obs(
                cmd, torch.as_tensor(last_act.reshape(-1, ACTION_DIM), device=device)
            )
            base_vel = lower.deterministic_action(lower_obs)
        else:
            base_vel = cmd  # flat policy outputs velocities directly
        return raw, cmd, base_vel, logp, nxt_hidden

    print(f"stage2_start algorithm={args.algorithm} agents={N} envs={n_envs} "
          f"lower_mode={lower_mode} device={device}", flush=True)

    while frames < args.total_steps:
        b_actor_obs, b_hidden, b_raw, b_lp = [], ([] if actor.use_memory else None), [], []
        b_critic_obs, b_rew, b_done, b_boot, b_val = [], [], [], [], []

        for _ in range(args.rollout_steps):
            exo_t = torch.as_tensor(exo.reshape(-1, EXO_OBS_DIM), dtype=torch.float32, device=device)
            if b_hidden is not None:
                b_hidden.append(hidden.detach().cpu().numpy())
            with timer.measure():  # full-hierarchy single-step inference time
                raw, cmd, base_vel, logp, next_hidden = full_inference(exo_t, hidden)

            joint = joint_of(exo)
            with torch.no_grad():
                values = critic(torch.as_tensor(joint, dtype=torch.float32, device=device))

            cmd_np = cmd.cpu().numpy().reshape(n_envs, N, -1)
            vel_np = base_vel.cpu().numpy().reshape(n_envs, N, ACTION_DIM)

            rewards = np.zeros(n_envs, dtype=np.float32)
            dones = np.zeros(n_envs, dtype=np.float32)
            boots = np.zeros(n_envs, dtype=np.float32)
            next_exo = exo.copy()
            for e, ad in enumerate(adapters):
                nx, rew, done, success, info = ad.step(vel_np[e], commands=cmd_np[e])
                rewards[e] = rew
                ep_returns[e] += rew
                last_arrived = float(info.get("arrived", last_arrived))
                if done:
                    dones[e] = 1.0
                    recent_returns.append(float(ep_returns[e]))
                    recent_success.append(1.0 if success else 0.0)
                    ep_returns[e] = 0.0
                    if not success:  # time-limit truncation -> bootstrap V(terminal)
                        with torch.no_grad():
                            tv = critic(torch.as_tensor(
                                nx.reshape(1, -1), dtype=torch.float32, device=device))
                        boots[e] = float(tv.item())
                    nx = ad.reset()
                    last_act[e] = 0.0
                    if next_hidden is not None:
                        next_hidden.view(n_envs, N, -1)[e].zero_()
                next_exo[e] = nx

            b_actor_obs.append(exo.reshape(-1, EXO_OBS_DIM).copy())
            b_raw.append(raw.cpu().numpy())
            b_lp.append(logp.cpu().numpy())
            b_critic_obs.append(joint.copy())
            b_rew.append(rewards.copy())
            b_done.append(dones.copy())
            b_boot.append(boots.copy())
            b_val.append(values.cpu().numpy())

            # roll forward (last_act only updates for non-reset envs; reset ones were zeroed)
            alive = dones < 0.5
            last_act[alive] = vel_np[alive]
            exo = next_exo
            hidden = None if next_hidden is None else next_hidden.detach()
            frames += n_envs

            if logger.should_log(frames):
                logger.log(frames, {
                    "reward": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                    "success_rate": float(np.mean(recent_success)) if recent_success else float("nan"),
                    "arrived": last_arrived,
                    **timer.stats(), **last_losses,
                })

        with torch.no_grad():
            last_value = critic(torch.as_tensor(
                joint_of(exo), dtype=torch.float32, device=device)).cpu().numpy()

        team_adv, team_ret = compute_gae_truncated(
            np.asarray(b_rew, np.float32), np.asarray(b_val, np.float32),
            np.asarray(b_done, np.float32), np.asarray(b_boot, np.float32),
            last_value, args.gamma, args.gae_lambda,
        )
        agent_adv = broadcast_team_advantage(team_adv, N)       # [T, E, N]

        adv_t = normalize_advantages(
            torch.as_tensor(agent_adv.reshape(-1), dtype=torch.float32, device=device))
        batch = MARLBatch(
            actor_obs=torch.as_tensor(
                np.asarray(b_actor_obs, np.float32), device=device).flatten(0, 1),
            actor_hidden=None if b_hidden is None else torch.as_tensor(
                np.asarray(b_hidden, np.float32), device=device).flatten(0, 1),
            raw_actions=torch.as_tensor(
                np.asarray(b_raw, np.float32), device=device).flatten(0, 1),
            old_log_probs=torch.as_tensor(
                np.asarray(b_lp, np.float32), device=device).flatten(0, 1),
            advantages=adv_t,
            critic_obs=torch.as_tensor(
                np.asarray(b_critic_obs, np.float32), device=device).flatten(0, 1),
            returns=torch.as_tensor(team_ret.reshape(-1), dtype=torch.float32, device=device),
        )
        losses = marl_ppo_update(
            actor, critic, optimizer, batch,
            clip_param=args.clip_param, entropy_coef=args.entropy_coef,
            ppo_epochs=args.ppo_epochs, num_minibatches=args.num_minibatches,
        )
        last_losses = {k: losses[k] for k in ("policy_loss", "value_loss", "entropy")}

        if frames >= next_ckpt:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "algorithm": args.algorithm, "seed": args.seed, "frames": frames,
                "actor": actor.state_dict(), "critic": critic.state_dict(),
                "n_agents": N, "lower_ckpt": args.lower_ckpt, "lower_mode": lower_mode,
            }, ckpt_dir / f"{args.algorithm}_seed_{args.seed}_{frames:09d}.pt")
            next_ckpt += args.ckpt_interval

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"{args.algorithm}_seed_{args.seed}_final.pt"
    torch.save({
        "algorithm": args.algorithm, "seed": args.seed, "frames": frames,
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "n_agents": N, "lower_ckpt": args.lower_ckpt, "lower_mode": lower_mode,
    }, final)
    print(f"stage2_done frames={frames} checkpoint={final}", flush=True)
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
