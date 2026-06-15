#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage 1.5: validate the FROZEN lower operator with MANUAL commands (人工指令).

The advisor's gate before any upper-layer learning: drive the frozen operator
with hand-crafted commands toward navigation goals.  "If the manual command
does not work, a learned/VLM command cannot work either."

    python dhrl_habitat/manual_validate.py \
        --lower-ckpt data/checkpoints/dhrl_repro/lower_velocity_seed_100.pt \
        --episodes 50

Reports: success rate, average steps-to-goal, average final distance, and the
single-step inference time (操作:命令生成+算子前向) -- the efficiency metric.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from habitat_commander.metrics import InferenceTimer  # noqa: E402

from dhrl_habitat.geometry import build_command  # noqa: E402
from dhrl_habitat.habitat_io import make_dhrl_env  # noqa: E402
from dhrl_habitat.layers import build_lower_obs, load_frozen_lower  # noqa: E402
from dhrl_habitat.multi_env import MultiRobotNavAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual-command validation of the frozen operator.")
    p.add_argument("--config-name", default="benchmark/multi_agent/hssd_spot_spot.yaml")
    p.add_argument("--project-dir", default=os.environ.get("PROJECT_DIR", str(Path.cwd())))
    p.add_argument("--lower-ckpt", required=True)
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--goal-mode", choices=["swap", "shared", "random"], default="swap")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--max-episode-steps", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    lower, mode = load_frozen_lower(args.lower_ckpt, device)
    print(f"loaded frozen lower operator: mode={mode} from {args.lower_ckpt}")

    action_keys = [f"agent_{i}_base_velocity" for i in range(args.n_agents)]
    env = make_dhrl_env(
        config_name=args.config_name, seed=args.seed,
        max_episode_steps=args.max_episode_steps, action_keys=action_keys,
    )
    adapter = MultiRobotNavAdapter(env, n_agents=args.n_agents, rng=rng, goal_mode=args.goal_mode)
    timer = InferenceTimer(device=str(device))

    last_action = np.zeros((args.n_agents, 3), dtype=np.float32)
    successes, steps_used, final_dists = [], [], []

    for ep in range(args.episodes):
        adapter.reset()
        last_action[:] = 0.0
        success, steps = False, 0
        for step in range(args.max_episode_steps):
            with timer.measure():  # manual command + frozen operator forward
                # waypoint_ego follows the NavMesh shortest path (straight-line
                # steering would run into walls rooms away from the goal).
                commands = np.stack(
                    [build_command(mode, *adapter.waypoint_ego(i)) for i in range(args.n_agents)]
                )
                lower_obs = build_lower_obs(
                    torch.as_tensor(commands, dtype=torch.float32, device=device),
                    torch.as_tensor(last_action, dtype=torch.float32, device=device),
                )
                base_vel = lower.deterministic_action(lower_obs).cpu().numpy()
            _, _, done, success, _info = adapter.step(base_vel, commands=commands)
            last_action = base_vel
            steps = step + 1
            if done:
                break
        successes.append(float(success))
        steps_used.append(steps)
        final_dists.append(float(np.mean(adapter._prev_geo)))
        print(f"episode {ep:03d}: success={success} steps={steps} "
              f"mean_final_geo={final_dists[-1]:.2f}m", flush=True)

    stats = timer.stats()
    print("\n==== manual-command validation (advisor gate) ====")
    print(f"episodes           : {args.episodes}")
    print(f"success rate       : {np.mean(successes) * 100:.1f}%")
    print(f"avg steps          : {np.mean(steps_used):.1f}")
    print(f"avg final geodesic : {np.mean(final_dists):.2f} m")
    print(f"inference per step : mean={stats['infer_ms_mean']:.3f} ms  "
          f"median={stats['infer_ms_median']:.3f} ms  p95={stats['infer_ms_p95']:.3f} ms")
    env.close()


if __name__ == "__main__":
    main()
