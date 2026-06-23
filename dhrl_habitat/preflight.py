#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Preflight: validate the new configs + adapters on the real Habitat install.

RUN THIS FIRST on AutoDL before any long training:

    python dhrl_habitat/preflight.py --config-name benchmark/multi_agent/hssd_spot_spot.yaml --n-agents 2

It (1) composes the config, (2) builds the env, (3) prints obs/action spaces,
(4) runs a short random + manual-command rollout through the adapter, and
(5) prints per-step displacement stats so you can calibrate
``nominal_step_disp`` / ``nominal_step_yaw`` in lower_env.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dhrl_habitat.geometry import build_command, ground_pos, heading_of, wrap_angle  # noqa: E402
from dhrl_habitat.habitat_io import make_dhrl_env  # noqa: E402
from dhrl_habitat.multi_env import MultiRobotNavAdapter  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="benchmark/multi_agent/replica_cad_spot_spot.yaml")
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()

    action_keys = [f"agent_{i}_base_velocity" for i in range(args.n_agents)]
    print(f"[1/5] composing + building env: {args.config_name}")
    env = make_dhrl_env(config_name=args.config_name, seed=args.seed,
                        max_episode_steps=200, action_keys=action_keys)
    print("      OK")

    print("[2/5] spaces:")
    print(f"      action_space = {env.action_space}")
    print(f"      obs keys     = {list(env.observation_space.spaces.keys())}")
    expected = 3 * args.n_agents
    actual = int(np.prod(env.action_space.shape))
    assert actual == expected, (
        f"action dim {actual} != {expected}: enable_lateral_move override failed?")
    print(f"      action dim   = {actual} (holonomic x {args.n_agents} agents)  OK")

    print("[3/5] adapter reset + exo obs:")
    adapter = MultiRobotNavAdapter(env, n_agents=args.n_agents,
                                   rng=np.random.default_rng(args.seed))
    exo = adapter.reset()
    print(f"      exo obs shape = {exo.shape}")
    print(f"      goals(xz)     = {adapter._goals_xz.tolist()}")
    print(f"      geo dists     = {adapter._prev_geo.tolist()}")

    print(f"[4/5] {args.steps} manual-command steps (forward displacement calibration):")
    prev_locs = [loc.copy() for loc in adapter._locs]
    disps, yaws = [], []
    for t in range(args.steps):
        cmds = np.stack([build_command("velocity", *adapter.waypoint_ego(i))
                         for i in range(args.n_agents)])  # also exercises find_path
        exo, rew, done, success, info = adapter.step(cmds, commands=cmds)
        for i in range(args.n_agents):
            d = ground_pos(adapter._locs[i]) - ground_pos(prev_locs[i])
            disps.append(float(np.hypot(*d)))
            yaws.append(abs(float(wrap_angle(
                heading_of(adapter._locs[i]) - heading_of(prev_locs[i])))))
        prev_locs = [loc.copy() for loc in adapter._locs]
        if t % 20 == 0:
            print(f"      t={t:03d} reward={rew:+.3f} arrived={info['arrived']} "
                  f"geo={[f'{g:.2f}' for g in adapter._prev_geo]}")
        if done:
            print(f"      episode ended at t={t} success={success}")
            exo = adapter.reset()
            prev_locs = [loc.copy() for loc in adapter._locs]

    print(f"[5/5] calibration: per-step displacement p95={np.percentile(disps, 95):.4f} m, "
          f"yaw p95={np.percentile(yaws, 95):.4f} rad")
    print("      -> set lower_env.py nominal_step_disp / nominal_step_yaw to ~these values")
    env.close()
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
