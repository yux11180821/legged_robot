#!/usr/bin/env python3
# A/B throughput + reward-sanity check for the geodesic-caching speedup in
# MultiRobotNavAdapter (geo_every=1 every-step vs geo_every=4 cached).
from __future__ import annotations
import os, sys, time, traceback
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dhrl_habitat.habitat_io import make_dhrl_env
from dhrl_habitat.multi_env import MultiRobotNavAdapter

KEYS = ["agent_0_base_velocity", "agent_1_base_velocity"]


def bench(geo_every: int):
    try:
        env = make_dhrl_env(config_name="benchmark/multi_agent/replica_cad_spot_spot.yaml",
                            seed=100, max_episode_steps=500, action_keys=KEYS)
        ad = MultiRobotNavAdapter(env, n_agents=2, rng=np.random.default_rng(7),
                                  goal_mode="swap", geo_every=geo_every)
        ad.reset()
        rng = np.random.default_rng(123)
        for _ in range(5):
            ad.step(rng.uniform(-0.5, 1.0, (2, 3)).astype(np.float32))
        n, tot_r, arrivals = 250, 0.0, 0
        t0 = time.perf_counter()
        for _ in range(n):
            o, r, d, s, info = ad.step(rng.uniform(-0.5, 1.0, (2, 3)).astype(np.float32))
            tot_r += r
            arrivals += int(info.get("arrived", 0) > 0)
            if d:
                ad.reset()
        dt = time.perf_counter() - t0
        print(f"RESULT geo_every={geo_every}  FPS={n/dt:.1f}  ({1000*dt/n:.1f} ms/step)  "
              f"mean_reward={tot_r/n:.3f}  steps_with_arrival={arrivals}", flush=True)
        env.close()
    except Exception:
        print(f"ERROR geo_every={geo_every}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    bench(1)
    bench(4)
    print("BENCH-DONE", flush=True)
