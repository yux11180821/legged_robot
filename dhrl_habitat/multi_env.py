# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage-2 adapter: N-robot cooperative navigation (corridor-crossing analog).

Task (paper's Corridor Crossing, indoor-scene edition): every robot must reach
its own goal; with ``goal_mode="swap"`` each robot's goal is another robot's
START position, which forces the robots to pass each other through the scene's
narrow corridors/doorways -- yield-or-rush coordination emerges exactly as in
the paper.  ``shared`` puts one common goal region; ``random`` is independent.

Per-agent observation e_t (PARTIAL, the paper's scalability design):
    [rel_goal_ego(2), goal_dist(1), nearest_neighbor_rel_ego(2), nn_dist(1),
     last_command(3)]                                  -> layers.EXO_OBS_DIM
Each agent sees only ITSELF + its NEAREST neighbor.  The joint observation for
the centralized critic is the concatenation over agents (training only).

Team reward (shared, paper Eq.1 homogeneous form):
    sum_i progress_i (geodesic)  + first-arrival bonus per agent
    + all-arrived bonus          - inter-agent collision penalty
    - proximity penalty          - slack
Episode ends on success (true terminal) or env time limit (truncation -> the
trainer bootstraps V(terminal), see marl_ppo/compute_gae_truncated).
"""

from __future__ import annotations

import numpy as np

from .geometry import ground_pos, rel_target_ego
from .habitat_io import collision_count_from_info, geodesic, get_sim, sample_navigable_point
from .layers import COMMAND_DIM, EXO_OBS_DIM


class MultiRobotNavAdapter:
    def __init__(
        self,
        gym_env,
        *,
        n_agents: int,
        rng: np.random.Generator,
        goal_mode: str = "swap",
        success_dist: float = 0.5,
        dist_norm: float = 10.0,
        progress_weight: float = 1.0,
        arrival_bonus: float = 2.5,
        all_arrived_bonus: float = 10.0,
        collision_penalty: float = 0.25,
        proximity_penalty: float = 0.05,
        proximity_dist: float = 0.8,
        slack: float = 0.005,
        geo_every: int = 1,
    ):
        assert goal_mode in ("swap", "shared", "random")
        self.env = gym_env
        self.n = n_agents
        self.rng = rng
        self.goal_mode = goal_mode
        self.success_dist = success_dist
        self.dist_norm = dist_norm
        self.w_prog = progress_weight
        self.b_arrive = arrival_bonus
        self.b_all = all_arrived_bonus
        self.w_coll = collision_penalty
        self.w_prox = proximity_penalty
        self.d_prox = proximity_dist
        self.slack = slack

        self.loc_keys = [f"agent_{i}_localization_sensor" for i in range(n_agents)]
        self.action_keys = [f"agent_{i}_base_velocity" for i in range(n_agents)]

        self._sim = None
        self._locs: list[np.ndarray] = []
        self._goals_xz: np.ndarray | None = None      # [N, 2]
        self._prev_geo: np.ndarray | None = None      # [N]
        self._arrived: np.ndarray | None = None       # [N] bool
        self._prev_coll_count = 0.0
        self._last_commands = np.zeros((n_agents, COMMAND_DIM), dtype=np.float32)
        self._base_height = 0.0
        self.geo_every = max(1, int(geo_every))
        self._step_count = 0
        self._cached_geo: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _ensure_sim(self):
        if self._sim is None:
            self._sim = get_sim(self.env)
        return self._sim

    def _read_locs(self, obs: dict) -> None:
        self._locs = [np.asarray(obs[k], dtype=np.float32) for k in self.loc_keys]
        self._base_height = float(np.mean([loc[1] for loc in self._locs]))

    def _sample_goals(self) -> None:
        starts = np.stack([ground_pos(loc) for loc in self._locs])  # [N,2]
        if self.goal_mode == "swap":
            # goal_i = start of agent (i+1) % N -> forced path crossing
            self._goals_xz = np.roll(starts, shift=-1, axis=0).copy()
        elif self.goal_mode == "shared":
            sim = self._ensure_sim()
            for _ in range(50):
                p = sample_navigable_point(sim, self.rng)
                g = np.array([p[0], p[2]], dtype=np.float32)
                if all(np.hypot(*(g - s)) >= 2.0 for s in starts):
                    break
            self._goals_xz = np.tile(g, (self.n, 1))
        else:  # random
            sim = self._ensure_sim()
            goals = []
            for i in range(self.n):
                for _ in range(50):
                    p = sample_navigable_point(sim, self.rng)
                    g = np.array([p[0], p[2]], dtype=np.float32)
                    if np.hypot(*(g - starts[i])) >= 2.0:
                        break
                goals.append(g)
            self._goals_xz = np.stack(goals)

    def _geo_dists(self) -> np.ndarray:
        sim = self._ensure_sim()
        return np.array(
            [
                geodesic(sim, ground_pos(self._locs[i]), self._goals_xz[i], height=self._base_height)
                for i in range(self.n)
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    def _agent_obs(self, i: int) -> np.ndarray:
        ego_f, ego_l, gdist = rel_target_ego(self._locs[i], self._goals_xz[i])
        # nearest neighbor (partial observation: ONLY the nearest one)
        p_i = ground_pos(self._locs[i])
        best_d, best_j = np.inf, -1
        for j in range(self.n):
            if j == i:
                continue
            d = float(np.hypot(*(ground_pos(self._locs[j]) - p_i)))
            if d < best_d:
                best_d, best_j = d, j
        nn_f, nn_l, nn_d = rel_target_ego(self._locs[i], ground_pos(self._locs[best_j]))
        out = np.array(
            [
                ego_f / self.dist_norm,
                ego_l / self.dist_norm,
                min(gdist, self.dist_norm) / self.dist_norm,
                nn_f / self.dist_norm,
                nn_l / self.dist_norm,
                min(nn_d, self.dist_norm) / self.dist_norm,
                *self._last_commands[i],
            ],
            dtype=np.float32,
        )
        return np.clip(out, -1.0, 1.0)

    def exo_obs(self) -> np.ndarray:
        """[N, EXO_OBS_DIM] per-agent partial observations."""
        out = np.stack([self._agent_obs(i) for i in range(self.n)])
        assert out.shape == (self.n, EXO_OBS_DIM)
        return out

    def joint_obs(self) -> np.ndarray:
        """[N * EXO_OBS_DIM] concatenation for the centralized critic."""
        return self.exo_obs().reshape(-1)

    def goal_ego(self, i: int) -> tuple[float, float, float]:
        """(ego_f, ego_l, dist) of agent i's goal -- for manual commands."""
        return rel_target_ego(self._locs[i], self._goals_xz[i])

    def waypoint_ego(self, i: int, lookahead: float = 1.5) -> tuple[float, float, float]:
        """Like goal_ego but steers along the NavMesh shortest path.

        Straight-line steering runs into walls in indoor scenes; the manual
        gate follows the next path waypoint instead.  Within ``lookahead`` of
        the goal it falls through to the goal itself (so stop_dist applies).
        """
        ego_f, ego_l, goal_dist = rel_target_ego(self._locs[i], self._goals_xz[i])
        if goal_dist < lookahead:
            return ego_f, ego_l, goal_dist
        from .habitat_io import next_waypoint

        wp = next_waypoint(
            self._ensure_sim(), ground_pos(self._locs[i]), self._goals_xz[i],
            height=self._base_height, lookahead=lookahead,
        )
        wf, wl, _ = rel_target_ego(self._locs[i], wp)
        return wf, wl, float(np.hypot(wf, wl))

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._read_locs(obs)
        self._sample_goals()
        self._prev_geo = self._geo_dists()
        self._arrived = np.zeros(self.n, dtype=bool)
        self._prev_coll_count = 0.0
        self._last_commands[:] = 0.0
        return self.exo_obs()

    def step(self, base_velocities: np.ndarray, commands: np.ndarray | None = None):
        """base_velocities: [N, 3] in (-1,1).  -> (exo, team_reward, done, success, info)

        ``commands`` (the ML output, [N,3]) is recorded into each agent's obs;
        for the no-hierarchy ablation pass the velocities themselves.
        """
        flat = np.asarray(base_velocities, dtype=np.float32).reshape(-1)
        obs, _, env_done, info = self.env.step(flat)
        self._read_locs(obs)
        self._last_commands = np.asarray(
            commands if commands is not None else base_velocities, dtype=np.float32
        ).reshape(self.n, COMMAND_DIM)

        cur_geo = self._geo_dists()
        progress = float(np.sum(self._prev_geo - cur_geo))
        self._prev_geo = cur_geo

        newly = (cur_geo < self.success_dist) & (~self._arrived)
        self._arrived |= cur_geo < self.success_dist
        success = bool(self._arrived.all())

        coll_count = collision_count_from_info(info)
        coll_delta = max(0.0, coll_count - self._prev_coll_count)
        self._prev_coll_count = coll_count

        prox_pairs = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.hypot(*(ground_pos(self._locs[i]) - ground_pos(self._locs[j]))))
                if d < self.d_prox:
                    prox_pairs += 1

        reward = (
            self.w_prog * progress
            + self.b_arrive * float(np.sum(newly))
            + (self.b_all if success else 0.0)
            - self.w_coll * coll_delta
            - self.w_prox * prox_pairs
            - self.slack
        )
        done = bool(env_done) or success
        info = dict(info)
        info.update(
            arrived=int(self._arrived.sum()),
            success=success,
            coll_delta=coll_delta,
        )
        return self.exo_obs(), float(reward), done, success, info
