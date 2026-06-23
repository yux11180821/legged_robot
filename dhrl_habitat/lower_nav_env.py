# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Lower-layer env adapter (corrected plan, 2026-06-17).

ONE Habitat env, TWO Spots:
  * agent_0 = the LEARNED policy.  A single-agent PPO learns its holonomic base
    velocity to perform point-goal navigation (reach its own sampled goal).
  * agent_1 = a SCRIPTED other agent.  It walks to its own goal along the
    NavMesh shortest path via a hand-coded velocity rule (the advisor-sanctioned
    "oracle-nav / fixed-velocity" baseline), and is given a fresh goal whenever
    it arrives so it stays a moving obstacle for agent_0 to avoid.

This realizes the spec exactly: "用 PPO 输出 1 个 Spot 的 action，每次只训练一个，
其他智能体用预设 policy 或规则".  The reward is agent_0's OWN navigation, so the
policy trains one Spot at a time.  Once trained it is frozen and the upper MAPPO
layer commands it.

Observation (agent_0, 6-D, all in [-1,1]):
    [goal_ego_forward, goal_ego_lateral, goal_dist,
     other_ego_forward, other_ego_lateral, other_dist]   (distances / dist_norm)

Reward (agent_0, single-agent point-goal):
    progress (geodesic decrease toward its goal)
    + arrival bonus (once)            - inter-agent collision penalty
    - proximity penalty (too close)   - slack
Episode ends on agent_0 arrival (TRUE terminal) or env time limit (TRUNCATION ->
the trainer bootstraps V(term_obs); see ppo.compute_gae_truncated).
"""

from __future__ import annotations

import numpy as np

from .geometry import ground_pos, manual_velocity_command, rel_target_ego
from .habitat_io import (
    collision_count_from_info,
    geodesic,
    get_sim,
    next_waypoint,
    sample_navigable_point,
)

OBS_DIM = 6
ACT_DIM = 3


class LowerNavAdapter:
    def __init__(
        self,
        gym_env,
        *,
        rng: np.random.Generator,
        success_dist: float = 0.5,
        dist_norm: float = 10.0,
        min_goal_dist: float = 3.0,
        progress_weight: float = 1.0,
        arrival_bonus: float = 10.0,
        collision_penalty: float = 0.25,
        proximity_penalty: float = 0.05,
        proximity_dist: float = 0.8,
        slack: float = 0.01,
        other_lookahead: float = 1.5,
    ):
        self.env = gym_env
        self.rng = rng
        self.success_dist = success_dist
        self.dist_norm = dist_norm
        self.min_goal_dist = min_goal_dist
        self.w_prog = progress_weight
        self.b_arrive = arrival_bonus
        self.w_coll = collision_penalty
        self.w_prox = proximity_penalty
        self.d_prox = proximity_dist
        self.slack = slack
        self.other_lookahead = other_lookahead

        self.loc_keys = ["agent_0_localization_sensor", "agent_1_localization_sensor"]
        self._sim = None
        self._locs: list[np.ndarray] = []
        self._base_height = 0.0
        self._goals_xz = np.zeros((2, 2), dtype=np.float32)  # [agent_0, agent_1]
        self._prev_geo0 = 0.0
        self._arrived = False
        self._prev_coll_count = 0.0

    # ------------------------------------------------------------------ #
    def _ensure_sim(self):
        if self._sim is None:
            self._sim = get_sim(self.env)
        return self._sim

    def _read_locs(self, obs: dict) -> None:
        self._locs = [np.asarray(obs[k], dtype=np.float32) for k in self.loc_keys]
        self._base_height = float(np.mean([loc[1] for loc in self._locs]))

    def _sample_goal_for(self, i: int) -> np.ndarray:
        sim = self._ensure_sim()
        start = ground_pos(self._locs[i])
        g = start
        for _ in range(50):
            p = sample_navigable_point(sim, self.rng)
            g = np.array([p[0], p[2]], dtype=np.float32)
            if np.hypot(*(g - start)) >= self.min_goal_dist:
                break
        return g

    def _sample_goals(self) -> None:
        self._goals_xz = np.stack([self._sample_goal_for(0), self._sample_goal_for(1)])

    def _geo(self, i: int) -> float:
        return geodesic(
            self._ensure_sim(), ground_pos(self._locs[i]), self._goals_xz[i],
            height=self._base_height,
        )

    def _obs(self) -> np.ndarray:
        gf, gl, gd = rel_target_ego(self._locs[0], self._goals_xz[0])
        nf, nl, nd = rel_target_ego(self._locs[0], ground_pos(self._locs[1]))
        out = np.array(
            [
                gf / self.dist_norm,
                gl / self.dist_norm,
                min(gd, self.dist_norm) / self.dist_norm,
                nf / self.dist_norm,
                nl / self.dist_norm,
                min(nd, self.dist_norm) / self.dist_norm,
            ],
            dtype=np.float32,
        )
        return np.clip(out, -1.0, 1.0)

    def _scripted_other_vel(self) -> np.ndarray:
        """agent_1: NavMesh-waypoint walk toward its goal -> holonomic velocity."""
        ego_f, ego_l, gd = rel_target_ego(self._locs[1], self._goals_xz[1])
        if gd >= self.other_lookahead:
            wp = next_waypoint(
                self._ensure_sim(), ground_pos(self._locs[1]), self._goals_xz[1],
                height=self._base_height, lookahead=self.other_lookahead,
            )
            ego_f, ego_l, _ = rel_target_ego(self._locs[1], wp)
            gd = float(np.hypot(ego_f, ego_l))
        return manual_velocity_command(ego_f, ego_l, gd)

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._read_locs(obs)
        self._sample_goals()
        self._prev_geo0 = self._geo(0)
        self._arrived = False
        self._prev_coll_count = 0.0
        return self._obs()

    def step(self, agent0_vel: np.ndarray):
        """agent0_vel: [3] in (-1,1).  -> (next_obs, reward, done, info).

        Auto-resets on done; ``info['truncated']`` flags a time-limit end (the
        trainer then bootstraps V(``info['term_obs']``)).
        """
        a0 = np.clip(np.asarray(agent0_vel, dtype=np.float32).reshape(3), -1.0, 1.0)
        a1 = self._scripted_other_vel()
        flat = np.concatenate([a0, a1]).astype(np.float32)
        obs, _, env_done, info = self.env.step(flat)
        self._read_locs(obs)

        # keep agent_1 moving: hand it a fresh goal once it arrives
        if self._geo(1) < self.success_dist:
            self._goals_xz[1] = self._sample_goal_for(1)

        cur_geo0 = self._geo(0)
        progress = self._prev_geo0 - cur_geo0
        self._prev_geo0 = cur_geo0

        newly_arrived = (cur_geo0 < self.success_dist) and (not self._arrived)
        self._arrived = self._arrived or (cur_geo0 < self.success_dist)

        coll_count = collision_count_from_info(info)
        coll_delta = max(0.0, coll_count - self._prev_coll_count)
        self._prev_coll_count = coll_count

        d_pair = float(np.hypot(*(ground_pos(self._locs[0]) - ground_pos(self._locs[1]))))
        prox = 1.0 if d_pair < self.d_prox else 0.0

        reward = (
            self.w_prog * progress
            + (self.b_arrive if newly_arrived else 0.0)
            - self.w_coll * coll_delta
            - self.w_prox * prox
            - self.slack
        )

        success = self._arrived
        done = bool(env_done) or success
        truncated = bool(env_done) and not success
        term_obs = self._obs()
        next_obs = self.reset() if done else term_obs
        out_info = dict(
            truncated=truncated, success=bool(success),
            term_obs=term_obs, geo=cur_geo0, coll_delta=coll_delta,
        )
        return next_obs, float(reward), done, out_info
