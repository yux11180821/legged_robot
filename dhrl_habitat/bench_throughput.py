#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Throughput diagnosis: how much does stripping the (unused) depth render cost?

The training obs is localization-only, but the agents still carry a
``depth_head_agent`` camera that Habitat renders every step.  This measures
env.step FPS with the depth sensors ON vs cleared, plus the cost of one
geodesic_distance call -- to find the real bottleneck before the 4-seed runs.
"""
from __future__ import annotations
import time
import traceback

import numpy as np
import habitat
from habitat.config import read_write
from habitat.gym import make_gym_from_config

ACTION_KEYS = ["agent_0_base_velocity", "agent_1_base_velocity"]


def build(strip_depth: bool):
    cfg = habitat.get_config("benchmark/multi_agent/replica_cad_spot_spot.yaml",
                             overrides=["habitat.environment.max_episode_steps=2000"])
    with read_write(cfg):
        cfg.habitat.gym.action_keys = ACTION_KEYS
        for k in ACTION_KEYS:
            ac = cfg.habitat.task.actions.get(k)
            if ac is not None and hasattr(ac, "enable_lateral_move"):
                ac.enable_lateral_move = True
        if strip_depth:
            for ag in list(cfg.habitat.simulator.agents.keys()):
                cfg.habitat.simulator.agents[ag].sim_sensors = {}
    return make_gym_from_config(cfg)


def bench(strip: bool):
    try:
        env = build(strip)
        env.reset()
        print(f"[strip={strip}] obs_keys={list(env.observation_space.spaces.keys())}", flush=True)
        a = np.zeros(int(np.prod(env.action_space.shape)), np.float32)
        a[0] = 0.4
        for _ in range(10):
            env.step(a)
        n = 150
        t0 = time.perf_counter()
        for _ in range(n):
            o, r, d, i = env.step(a)
            if d:
                env.reset()
        dt = time.perf_counter() - t0
        print(f"RESULT strip_depth={strip}  FPS={n/dt:.1f}  ({1000*dt/n:.1f} ms/step)", flush=True)
        env.close()
    except Exception:
        print(f"ERROR strip={strip}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    bench(False)
    bench(True)
    print("BENCH-DONE", flush=True)
