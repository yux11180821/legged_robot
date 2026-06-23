#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""UPPER LAYER: MAPPO that outputs COMMANDS to the FROZEN lower nav policy.

Corrected plan (2026-06-17), stage 2.  The lower layer (``lower_nav_train.py``)
is a single-agent PPO point-goal navigator that was trained one Spot at a time
and is now FROZEN.  Here a multi-agent MAPPO policy (CTDE: shared actor +
centralized critic, the paper's homogeneous Eq.1 form) commands several of these
frozen lowers to cooperate.

Hierarchy wiring (the key design):
  * The frozen lower's observation is [goal_ego(3), other_ego(3)] -- it learned
    to drive its base velocity toward the goal encoded by goal_ego while reactively
    avoiding the other agent encoded by other_ego.
  * The UPPER's command (3-D, in (-1,1)) is a *virtual goal_ego* -- a "carrot"
    the upper places in the agent's ego frame each step.  We feed the lower
    [command(3), other_ego(3)] (other_ego read from the live env state), so:
        lower_obs = concat(upper_command, nearest_neighbor_ego)
        base_velocity = frozen_lower(lower_obs)
  * The upper learns WHERE to place the carrot to coordinate (yield in corridors,
    route around each other); the lower handles goal-reaching + local avoidance.

``other_ego`` is exactly ``MultiRobotNavAdapter.exo_obs[..., 3:6]`` (nn ego,
already /dist_norm-normalized to the lower's training scale), so no extra plumbing.

    python dhrl_habitat/upper_mappo_train.py --algorithm dhrl --seed 100 \
        --lower-ckpt data/checkpoints/lower_nav/lower_nav_seed_100_final.pt

Ablations (paper Fig.5 / Table 1) come free from ``build_actor``:
    dhrl          UL + ML(GRU) + frozen LL    (the proposed method, default)
    no_memory     UL + ML(no GRU) + frozen LL (No Spatiotemporal Memory)
    no_hierarchy  flat MLP -> base velocity   (No Hierarchy; --lower-ckpt unused)
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
from habitat_commander.ppo import normalize_advantages  # noqa: E402

from dhrl_habitat.habitat_io import make_dhrl_env  # noqa: E402
from dhrl_habitat.layers import (  # noqa: E402
    ACTION_DIM, ALGORITHMS, EXO_OBS_DIM, CentralCritic, build_actor,
)
from dhrl_habitat.lower_nav_env import OBS_DIM as LOWER_OBS_DIM  # 6  # noqa: E402
from dhrl_habitat.marl_ppo import (  # noqa: E402
    MARLBatch, broadcast_team_advantage, compute_gae_truncated, marl_ppo_update,
)
from dhrl_habitat.multi_env import MultiRobotNavAdapter  # noqa: E402

# Slice of MultiRobotNavAdapter.exo_obs that equals the lower's other_ego obs.
_NN_EGO_SLICE = slice(3, 6)


def load_frozen_lower_nav(path: str, device: torch.device) -> CommanderPolicy:
    """Load the stage-1 single-agent nav policy and FREEZE it (paper's constraint)."""
    payload = torch.load(path, map_location=device)
    lower = CommanderPolicy(
        obs_dim=payload.get("obs_dim", LOWER_OBS_DIM),
        action_dim=payload.get("action_dim", ACTION_DIM),
        hidden_size=payload.get("hidden_size", 256),
        use_memory=False,
    )
    lower.load_state_dict(payload["policy"])
    lower.to(device).eval()
    for p in lower.parameters():
        p.requires_grad_(False)
    return lower


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 upper MAPPO over the frozen nav lower.")
    p.add_argument("--config-name", default="benchmark/multi_agent/replica_cad_spot_spot.yaml")
    p.add_argument("--exp-name", default="upper_mappo")
    p.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(Path.cwd())))
    p.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="dhrl")
    p.add_argument("--lower-ckpt", default=None,
                   help="Frozen stage-1 nav policy (required for dhrl / no_memory).")
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--goal-mode", choices=["swap", "shared", "random"], default="swap")
    p.add_argument("--geo-every", type=int, default=4,
                   help="Recompute geodesic team-reward every K steps (telescoping-equivalent; "
                        "true geodesic kept for arrival/success). K=1 disables the speedup.")
    p.add_argument("--no-render", action="store_true",
                   help="Clear depth cameras (obs is localization-only) -> no per-step GPU "
                        "render -> removes GPU contention when many envs run concurrently.")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)        # paper: Adam 5e-4
    p.add_argument("--gamma", type=float, default=0.995)    # paper: 0.995
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=1e-3)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-interval", type=int, default=25_000)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_envs = 1
        args.total_steps = 256
        args.rollout_steps = 16
        args.max_episode_steps = 60
        args.log_every = 32
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ALGORITHMS[args.algorithm]

    lower = None
    if spec.hierarchical:
        if not args.lower_ckpt:
            raise SystemExit(f"--lower-ckpt is required for algorithm={args.algorithm}")
        lower = load_frozen_lower_nav(args.lower_ckpt, device)

    action_keys = [f"agent_{i}_base_velocity" for i in range(args.n_agents)]
    envs, adapters = [], []
    for i in range(max(1, args.num_envs)):
        env = make_dhrl_env(
            config_name=args.config_name, seed=args.seed + i,
            max_episode_steps=args.max_episode_steps, action_keys=action_keys,
            strip_render=args.no_render,
        )
        envs.append(env)
        adapters.append(MultiRobotNavAdapter(
            env, n_agents=args.n_agents, rng=np.random.default_rng(args.seed + i),
            goal_mode=args.goal_mode, geo_every=args.geo_every,
        ))
    n_envs, N = len(envs), args.n_agents
    joint_dim = N * EXO_OBS_DIM

    actor, _ = build_actor(args.algorithm, exo_dim=EXO_OBS_DIM, hidden_size=args.hidden_size)
    actor.to(device)
    critic = CentralCritic(joint_dim, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5)

    results_dir = Path(args.project_dir) / "results" / args.exp_name
    ckpt_dir = Path(args.project_dir) / "data" / "checkpoints" / args.exp_name
    logger = StepLogger(
        results_dir / f"{args.algorithm}_seed_{args.seed}_metrics.csv", log_every=args.log_every,
        fields=["env_steps", "reward", "success_rate", "arrived",
                "infer_ms_mean", "infer_ms_median", "infer_ms_p95", "infer_ms_last",
                "policy_loss", "value_loss", "entropy"])
    timer = InferenceTimer(device=str(device))

    exo = np.stack([ad.reset() for ad in adapters])            # [E, N, exo]
    hidden = actor.initial_hidden(n_envs * N, device)
    frames, next_ckpt = 0, args.ckpt_interval
    ep_returns = np.zeros(n_envs, dtype=np.float32)
    recent_returns: deque[float] = deque(maxlen=50)
    recent_success: deque[float] = deque(maxlen=50)
    last_losses: dict[str, float] = {}
    last_arrived = 0.0

    def joint_of(exo_arr: np.ndarray) -> np.ndarray:
        return exo_arr.reshape(exo_arr.shape[0], -1)

    def full_inference(exo_t: torch.Tensor, hid):
        """exo_t: [E*N, exo].  -> raw, cmd, base_velocity, logp, next_hidden.

        The upper command IS the virtual goal_ego fed to the frozen lower; the
        lower's other_ego comes from the live exo obs slice [3:6].
        """
        raw, cmd, logp, nxt_hidden = actor.act(exo_t, hid)
        if spec.hierarchical:
            # The command is a virtual goal_ego "carrot" for the frozen lower.
            # The lower was trained with goal_ego = [f/10, l/10, hypot(f,l)/10]:
            # a direction PLUS its own ALWAYS-POSITIVE magnitude, with the 3rd
            # slot == hypot of the first two.  Feeding the raw 3-D tanh command
            # verbatim is out-of-distribution (its 3rd slot can be negative and
            # is decoupled from the first two), which can stall the frozen lower.
            # So use cmd[0:2] as the ego direction and SYNTHESIZE the distance
            # slot = hypot(cmd0, cmd1) -> in-distribution.  cmd[2] is left unused.
            carrot = cmd[:, :2]
            dist = torch.linalg.vector_norm(carrot, dim=-1, keepdim=True)
            lower_obs = torch.cat([carrot, dist, exo_t[:, _NN_EGO_SLICE]], dim=-1)  # [E*N, 6]
            base_vel = lower.deterministic_action(lower_obs, None)
        else:
            base_vel = cmd
        return raw, cmd, base_vel, logp, nxt_hidden

    print(f"upper_mappo start algo={args.algorithm} agents={N} envs={n_envs} "
          f"hierarchical={spec.hierarchical} device={device}", flush=True)

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

            exo = next_exo
            hidden = None if next_hidden is None else next_hidden.detach()
            frames += n_envs

            if logger.should_log(frames):
                logger.log(frames, {
                    "reward": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                    "success_rate": float(np.mean(recent_success)) if recent_success else float("nan"),
                    "arrived": last_arrived, **timer.stats(), **last_losses})

        with torch.no_grad():
            last_value = critic(torch.as_tensor(
                joint_of(exo), dtype=torch.float32, device=device)).cpu().numpy()

        team_adv, team_ret = compute_gae_truncated(
            np.asarray(b_rew, np.float32), np.asarray(b_val, np.float32),
            np.asarray(b_done, np.float32), np.asarray(b_boot, np.float32),
            last_value, args.gamma, args.gae_lambda)
        agent_adv = broadcast_team_advantage(team_adv, N)

        adv_t = normalize_advantages(
            torch.as_tensor(agent_adv.reshape(-1), dtype=torch.float32, device=device))
        batch = MARLBatch(
            actor_obs=torch.as_tensor(np.asarray(b_actor_obs, np.float32), device=device).flatten(0, 1),
            actor_hidden=None if b_hidden is None else torch.as_tensor(
                np.asarray(b_hidden, np.float32), device=device).flatten(0, 1),
            raw_actions=torch.as_tensor(np.asarray(b_raw, np.float32), device=device).flatten(0, 1),
            old_log_probs=torch.as_tensor(np.asarray(b_lp, np.float32), device=device).flatten(0, 1),
            advantages=adv_t,
            critic_obs=torch.as_tensor(np.asarray(b_critic_obs, np.float32), device=device).flatten(0, 1),
            returns=torch.as_tensor(team_ret.reshape(-1), dtype=torch.float32, device=device))
        if not args.no_anneal_lr:
            lr_now = args.lr * max(0.0, 1.0 - frames / args.total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
        losses = marl_ppo_update(
            actor, critic, optimizer, batch,
            clip_param=args.clip_param, entropy_coef=args.entropy_coef,
            ppo_epochs=args.ppo_epochs, num_minibatches=args.num_minibatches,
            target_kl=(args.target_kl if args.target_kl > 0 else None))
        last_losses = {k: losses[k] for k in ("policy_loss", "value_loss", "entropy")}

        if frames >= next_ckpt:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"algorithm": args.algorithm, "seed": args.seed, "frames": frames,
                        "actor": actor.state_dict(), "critic": critic.state_dict(),
                        "n_agents": N, "lower_ckpt": args.lower_ckpt},
                       ckpt_dir / f"{args.algorithm}_seed_{args.seed}_{frames:09d}.pt")
            next_ckpt += args.ckpt_interval

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"{args.algorithm}_seed_{args.seed}_final.pt"
    torch.save({"algorithm": args.algorithm, "seed": args.seed, "frames": frames,
                "actor": actor.state_dict(), "critic": critic.state_dict(),
                "n_agents": N, "lower_ckpt": args.lower_ckpt}, final)
    print(f"upper_mappo done frames={frames} checkpoint={final}", flush=True)
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
