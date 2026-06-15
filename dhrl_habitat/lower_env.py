# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage-1 adapter: single-agent training of the lower operator (算子).

Uses the same hssd_spot_spot config but controls ONLY agent_0 (agent_1 receives
no actions and stands still -- the gym wrapper drops actions not in
action_keys).  Two control modes, both from the paper (§5.2.1):

  position : a target waypoint is sampled near the agent; the command is the
             CURRENT relative target in the ego frame (normalized by radius).
             Reward = reciprocal L2 distance  1/(1+d)  + success bonus, then a
             new target is sampled (continuing episode).
  velocity : a target velocity (vx, vy, wz) is sampled and held for a while;
             reward = dot(actual ego velocity, target velocity) (paper's
             velocity-mode reward), where actual velocity is estimated from the
             per-step ego displacement, + a yaw-rate tracking term for wz.

The lower-layer observation is ALWAYS [command(3), last_action(3)] -- identical
to what the frozen operator sees in stage 2 (built by layers.build_lower_obs).
"""

from __future__ import annotations

import numpy as np

from .geometry import (
    ground_pos,
    heading_of,
    manual_position_command,
    rel_target_ego,
    wrap_angle,
)
from .habitat_io import get_sim, sample_navigable_point


class LowerTrainAdapter:
    """Wraps one gym env (agent_0 only) into the stage-1 training interface."""

    def __init__(
        self,
        gym_env,
        *,
        mode: str = "velocity",
        rng: np.random.Generator,
        loc_key: str = "agent_0_localization_sensor",
        success_dist: float = 0.3,
        waypoint_radius: float = 2.0,
        target_min_dist: float = 1.0,
        target_max_dist: float = 4.0,
        vel_hold_steps: int = 50,
        success_bonus: float = 5.0,
        # per-step ego displacement / yaw at full command, used to normalize the
        # velocity-mode rewards to ~O(1).  Defaults = lin/ang_speed (10) divided
        # by habitat's default ctrl_freq (120 Hz integration dt) = 0.0833 m|rad
        # per step.  Verify against preflight.py's measured p95 on AutoDL.
        nominal_step_disp: float = 10.0 / 120.0,
        nominal_step_yaw: float = 10.0 / 120.0,
    ):
        assert mode in ("position", "velocity")
        self.env = gym_env
        self.mode = mode
        self.rng = rng
        self.loc_key = loc_key
        self.success_dist = success_dist
        self.radius = waypoint_radius
        self.target_min = target_min_dist
        self.target_max = target_max_dist
        self.vel_hold_steps = vel_hold_steps
        self.success_bonus = success_bonus
        self.nominal_disp = nominal_step_disp
        self.nominal_yaw = nominal_step_yaw

        self._sim = None
        self._loc = None
        self._target_xz: np.ndarray | None = None
        self._target_vel: np.ndarray | None = None
        self._steps_since_target = 0
        self.last_action = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _ensure_sim(self):
        if self._sim is None:
            self._sim = get_sim(self.env)
        return self._sim

    def _sample_position_target(self) -> None:
        """Navigable waypoint between target_min and target_max from the agent."""
        p = ground_pos(self._loc)
        for _ in range(50):
            cand = sample_navigable_point(self._ensure_sim(), self.rng)
            d = float(np.hypot(cand[0] - p[0], cand[2] - p[1]))
            if self.target_min <= d <= self.target_max:
                self._target_xz = np.array([cand[0], cand[2]], dtype=np.float32)
                return
        # fall back to whatever was sampled last
        self._target_xz = np.array([cand[0], cand[2]], dtype=np.float32)

    def _sample_velocity_target(self) -> None:
        # 10% zero "stop" commands: the upper layer / manual gate emits zeros
        # near the goal, so the frozen operator must have seen them in training.
        if self.rng.random() < 0.1:
            self._target_vel = np.zeros(3, dtype=np.float32)
            self._steps_since_target = 0
            return
        v = self.rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        # avoid near-zero commands (nothing to track)
        if np.linalg.norm(v[:2]) < 0.3:
            v[0] = np.sign(v[0] + 1e-6) * 0.5
        self._target_vel = np.clip(v, -1.0, 1.0)
        self._steps_since_target = 0

    # ------------------------------------------------------------------ #
    def _command(self) -> np.ndarray:
        if self.mode == "velocity":
            return self._target_vel.copy()
        ego_f, ego_l, dist = rel_target_ego(self._loc, self._target_xz)
        return manual_position_command(ego_f, ego_l, dist, radius=self.radius, stop_dist=0.0)

    def lower_obs(self) -> np.ndarray:
        """[command(3), last_action(3)] -- the operator's proprio obs p_t."""
        return np.concatenate([self._command(), self.last_action]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._loc = np.asarray(obs[self.loc_key], dtype=np.float32)
        self.last_action[:] = 0.0
        if self.mode == "position":
            self._sample_position_target()
        else:
            self._sample_velocity_target()
        return self.lower_obs()

    def step(self, action: np.ndarray):
        """action = base velocity in (-1,1)^3 -> (obs, reward, done, info)."""
        prev_loc = self._loc.copy()
        obs, _, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self._loc = np.asarray(obs[self.loc_key], dtype=np.float32)
        self.last_action = np.asarray(action, dtype=np.float32).copy()

        if self.mode == "position":
            reward, resample = self._position_reward()
            if resample:
                self._sample_position_target()
        else:
            reward = self._velocity_reward(prev_loc)
            self._steps_since_target += 1
            if self._steps_since_target >= self.vel_hold_steps:
                self._sample_velocity_target()

        return self.lower_obs(), float(reward), bool(done), info

    # ------------------------------------------------------------------ #
    def _position_reward(self) -> tuple[float, bool]:
        _, _, dist = rel_target_ego(self._loc, self._target_xz)
        reward = 1.0 / (1.0 + dist)  # paper: reciprocal L2 distance
        if dist < self.success_dist:
            return reward + self.success_bonus, True
        return reward, False

    def _velocity_reward(self, prev_loc: np.ndarray) -> float:
        # actual per-step ego displacement (in the PREVIOUS ego frame)
        p0, p1 = ground_pos(prev_loc), ground_pos(self._loc)
        h0 = heading_of(prev_loc)
        d = p1 - p0
        from .geometry import world_to_ego

        ego_f, ego_l = world_to_ego(float(d[0]), float(d[1]), h0)
        v_act = np.array([ego_f, ego_l], dtype=np.float32) / self.nominal_disp
        lin = float(np.dot(v_act, self._target_vel[:2]))  # paper: velocity dot product
        dyaw = float(wrap_angle(heading_of(self._loc) - h0))
        ang = -0.25 * abs(dyaw / self.nominal_yaw - float(self._target_vel[2]))
        return lin + ang
