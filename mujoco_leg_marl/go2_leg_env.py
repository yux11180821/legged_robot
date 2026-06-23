# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Per-leg MARL locomotion env for Unitree Go2 in MuJoCo (the LOWER layer).

One Go2 -> FOUR agents (FL / FR / RL / RR), each controls its leg's 3 joints.
This is the wheel-legged-transfer target's lower layer: the paper's distributed
hierarchical idea applied at the LEG level (each leg an agent) rather than the
robot level.

  * decentralized execution: each agent's actor sees ONLY its leg's
    proprioception + shared body IMU (base lin/ang vel, projected gravity) + the
    velocity command.  No vision.
  * CTDE: a centralized critic reads the full robot state (training only).
  * action = per-leg joint position targets (offset from the home stance),
    converted to torque by an in-Python PD controller (Go2 menagerie actuators
    are torque motors with no built-in PD).
  * reward = SHARED balance/velocity-tracking (posture-balance shared reward)
    broadcast to all 4 legs + PER-LEG energy/smoothness (each leg minimises its
    own torque/jerk).
  * velocity command = the interface the future upper layer (Habitat Commander)
    will drive; here it is randomly sampled per episode.

All model facts verified against mujoco_menagerie/unitree_go2 (scene.xml):
12 joints <leg>_<hip|thigh|calf>_joint, torque actuators (ctrlrange = torque
limit), foot geoms FL/FR/RL/RR on the calf bodies, base body "base", home
keyframe qpos legs=[0,0.9,-1.8] base z=0.27, dt=0.002, NO sensors (IMU/contact
computed from qpos/qvel/contact).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINTS_PER_LEG = 3
NUM_JOINTS = JOINTS_PER_LEG * len(LEG_NAMES)  # 12

# per-leg obs: base_lin_b(3) base_ang_b(3) proj_grav(3) cmd(3) jpos_rel(3) jvel(3) foot(1) last_act(3)
PER_LEG_OBS_DIM = 3 + 3 + 3 + 3 + JOINTS_PER_LEG + JOINTS_PER_LEG + 1 + JOINTS_PER_LEG  # 22
# central state: base_lin_b(3) base_ang_b(3) grav(3) cmd(3) jpos_rel(12) jvel(12) foot(4) last_act(12)
STATE_DIM = 3 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + len(LEG_NAMES) + NUM_JOINTS  # 52


@dataclass
class Go2LegMARLCfg:
    model_path: str = "mujoco_menagerie/unitree_go2/scene.xml"
    num_envs: int = 32
    decimation: int = 10          # dt=0.002 -> 50 Hz control
    episode_length_s: float = 20.0
    # PD (Go2 actuators are torque motors; we add PD in Python)
    kp: float = 25.0
    kd: float = 0.5
    action_scale: float = 0.3
    # obs scales
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 0.25
    joint_vel_scale: float = 0.05
    # velocity command ranges (the Commander interface) -- forward-biased
    # forward-biased, reduced range -> easier "walk forward" task to learn first
    cmd_lin_x: tuple[float, float] = (0.0, 1.0)
    cmd_lin_y: tuple[float, float] = (-0.3, 0.3)
    cmd_ang_z: tuple[float, float] = (-0.5, 0.5)
    cmd_resample_s: float = 5.0
    external_commands: bool = False  # True -> set_velocity_commands() drives it (upper layer)
    # shared reward (legged_gym-style: velocity tracking + gait dominate; NO big
    # alive bonus -- a constant alive reward makes "stand still" the optimum).
    tracking_sigma: float = 0.25
    rew_track_lin: float = 2.0
    rew_track_ang: float = 0.75
    pen_lin_vel_z: float = -2.0
    pen_ang_vel_xy: float = -0.05
    pen_flat: float = -2.5
    base_height_target: float = 0.30
    pen_base_height: float = -1.0          # softer (body bobs while walking)
    rew_alive: float = 0.0                  # removed: was the "stand still" attractor
    rew_termination: float = -50.0          # keeps it upright without rewarding standing
    # forward-progress: dense reward for base velocity along the command direction
    # (gradient everywhere + must actually translate -> stepping must emerge to score)
    rew_progress: float = 1.0
    # gait: feet air-time -- DISABLED (got gamed as "shuffle in place"); rely on progress
    rew_feet_air_time: float = 0.0
    feet_air_time_target: float = 0.4
    cmd_active_thresh: float = 0.1
    # per-leg reward
    pen_torque: float = -2.0e-4
    pen_action_rate: float = -0.01
    # termination
    term_min_height: float = 0.18
    term_grav_z: float = -0.5     # projected_gravity z > this => tilted past ~60 deg

    possible_agents: list[str] = field(default_factory=lambda: list(LEG_NAMES))


class Go2LegMARLEnv:
    """Vectorized (N parallel MjData) per-leg MARL env. Pure-CPU mj_step."""

    def __init__(self, cfg: Go2LegMARLCfg):
        self.cfg = cfg
        self.n = cfg.num_envs
        self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.model.opt.timestep = 0.002
        self.datas = [mujoco.MjData(self.model) for _ in range(self.n)]
        self.dt = self.model.opt.timestep * cfg.decimation
        self.max_steps = int(round(cfg.episode_length_s / self.dt))
        self.cmd_resample = max(1, int(round(cfg.cmd_resample_s / self.dt)))
        self.rng = np.random.default_rng()

        m = self.model
        # per-leg index maps (resolved by name -> robust to model ordering)
        self.leg_qpos = {}
        self.leg_dof = {}
        self.leg_act = {}
        self.foot_gid = {}
        for leg in LEG_NAMES:
            js = [f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint"]
            acts = [f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf"]
            self.leg_qpos[leg] = np.array([m.joint(j).qposadr[0] for j in js], dtype=np.int64)
            self.leg_dof[leg] = np.array([m.joint(j).dofadr[0] for j in js], dtype=np.int64)
            self.leg_act[leg] = np.array([m.actuator(a).id for a in acts], dtype=np.int64)
            self.foot_gid[leg] = m.geom(leg).id
        self.all_qpos = np.concatenate([self.leg_qpos[l] for l in LEG_NAMES])  # 12
        self.all_dof = np.concatenate([self.leg_dof[l] for l in LEG_NAMES])
        self.all_act = np.concatenate([self.leg_act[l] for l in LEG_NAMES])
        self.base_id = m.body("base").id
        self.home_id = m.key("home").id
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self.home_qpos = m.key_qpos[self.home_id].copy()
        self.home_leg = self.home_qpos[self.all_qpos]  # 12 default joint angles

        # buffers
        self.commands = np.zeros((self.n, 3), dtype=np.float32)
        self.last_action = np.zeros((self.n, len(LEG_NAMES), JOINTS_PER_LEG), dtype=np.float32)
        self.prev_action = np.zeros_like(self.last_action)
        self.step_buf = np.zeros(self.n, dtype=np.int64)
        self._died = np.zeros(self.n, dtype=bool)
        # feet air-time gait tracking (per foot, seconds since leaving the ground)
        self.feet_air_time = np.zeros((self.n, len(LEG_NAMES)), dtype=np.float32)
        self.last_contact = np.ones((self.n, len(LEG_NAMES)), dtype=bool)

        assert PER_LEG_OBS_DIM == 22 and STATE_DIM == 52

    # ------------------------------------------------------------------ #
    def _resample_cmd(self, idx: np.ndarray) -> None:
        if self.cfg.external_commands:
            return
        c = self.cfg
        self.commands[idx, 0] = self.rng.uniform(*c.cmd_lin_x, len(idx))
        self.commands[idx, 1] = self.rng.uniform(*c.cmd_lin_y, len(idx))
        self.commands[idx, 2] = self.rng.uniform(*c.cmd_ang_z, len(idx))

    def set_velocity_commands(self, commands: np.ndarray) -> None:
        """Drive the (frozen) lower layer from an external upper-layer Commander."""
        self.commands[:] = np.asarray(commands, dtype=np.float32)

    def _reset_idx(self, idx: np.ndarray) -> None:
        for e in idx:
            d = self.datas[e]
            mujoco.mj_resetDataKeyframe(self.model, d, self.home_id)
            d.qpos[self.all_qpos] += self.rng.uniform(-0.1, 0.1, NUM_JOINTS)
            mujoco.mj_forward(self.model, d)
        self.step_buf[idx] = 0
        self.last_action[idx] = 0.0
        self.prev_action[idx] = 0.0
        self.feet_air_time[idx] = 0.0
        self.last_contact[idx] = True
        self._resample_cmd(idx)

    def reset(self) -> np.ndarray:
        self._reset_idx(np.arange(self.n))
        return self._observations()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _base_frame(d):
        """-> (lin_vel_body[3], ang_vel_body[3], projected_gravity[3], base_z)."""
        quat = d.qpos[3:7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        lin_b = R.T @ d.qvel[0:3]
        ang_b = d.qvel[3:6].copy()  # already body frame (gyro)
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])
        return lin_b.astype(np.float32), ang_b.astype(np.float32), proj_g.astype(np.float32), float(d.qpos[2])

    def _foot_contacts(self, d) -> np.ndarray:
        """[4] boolean (as float) foot-floor contact, leg order FL/FR/RL/RR."""
        out = np.zeros(len(LEG_NAMES), dtype=np.float32)
        gid2leg = {self.foot_gid[l]: i for i, l in enumerate(LEG_NAMES)}
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                if g in gid2leg:
                    out[gid2leg[g]] = 1.0
        return out

    def _apply_pd_and_step(self, actions: np.ndarray) -> None:
        """actions: [N, 4, 3] in (-1,1).  PD position-target -> torque, decimation steps."""
        c = self.cfg
        targets = self.home_leg[None, :] + c.action_scale * actions.reshape(self.n, NUM_JOINTS)
        for _ in range(c.decimation):
            for e in range(self.n):
                d = self.datas[e]
                q = d.qpos[self.all_qpos]
                qd = d.qvel[self.all_dof]
                tau = c.kp * (targets[e] - q) - c.kd * qd
                d.ctrl[self.all_act] = np.clip(tau, self.ctrl_lo[self.all_act], self.ctrl_hi[self.all_act])
                mujoco.mj_step(self.model, d)

    # ------------------------------------------------------------------ #
    def _observations(self):
        """-> (exo[N,4,22], state[N,52]) built from current sim state."""
        c = self.cfg
        exo = np.zeros((self.n, len(LEG_NAMES), PER_LEG_OBS_DIM), dtype=np.float32)
        state = np.zeros((self.n, STATE_DIM), dtype=np.float32)
        for e in range(self.n):
            d = self.datas[e]
            lin_b, ang_b, grav, _ = self._base_frame(d)
            foot = self._foot_contacts(d)
            jpos_rel = (d.qpos[self.all_qpos] - self.home_leg).astype(np.float32)
            jvel = d.qvel[self.all_dof].astype(np.float32) * c.joint_vel_scale
            cmd = self.commands[e]
            shared = np.concatenate([
                lin_b * c.lin_vel_scale, ang_b * c.ang_vel_scale, grav,
                cmd * np.array([c.lin_vel_scale, c.lin_vel_scale, c.ang_vel_scale], np.float32),
            ])  # 12
            for i, leg in enumerate(LEG_NAMES):
                sl = slice(i * JOINTS_PER_LEG, (i + 1) * JOINTS_PER_LEG)
                exo[e, i] = np.concatenate([
                    shared, jpos_rel[sl], jvel[sl], foot[i:i + 1], self.last_action[e, i],
                ])
            state[e] = np.concatenate([
                lin_b * c.lin_vel_scale, ang_b * c.ang_vel_scale, grav, cmd,
                jpos_rel, jvel, foot, self.last_action[e].reshape(-1),
            ])
        return exo, state

    def _rewards_dones(self):
        """-> (per_agent_reward[N,4], done[N], success[N]=False, info)."""
        c = self.cfg
        team = np.zeros(self.n, dtype=np.float32)
        leg_pen = np.zeros((self.n, len(LEG_NAMES)), dtype=np.float32)
        died = np.zeros(self.n, dtype=bool)
        contacts = np.zeros((self.n, len(LEG_NAMES)), dtype=np.float32)
        for e in range(self.n):
            d = self.datas[e]
            lin_b, ang_b, grav, base_z = self._base_frame(d)
            contacts[e] = self._foot_contacts(d)
            # shared: velocity tracking + posture balance
            lin_err = np.sum((self.commands[e, :2] - lin_b[:2]) ** 2)
            ang_err = (self.commands[e, 2] - ang_b[2]) ** 2
            r = c.rew_track_lin * np.exp(-lin_err / c.tracking_sigma)
            r += c.rew_track_ang * np.exp(-ang_err / c.tracking_sigma)
            r += c.pen_lin_vel_z * lin_b[2] ** 2
            r += c.pen_ang_vel_xy * np.sum(ang_b[:2] ** 2)
            r += c.pen_flat * np.sum(grav[:2] ** 2)
            r += c.pen_base_height * (base_z - c.base_height_target) ** 2
            r += c.rew_progress * float(np.clip(np.dot(lin_b[:2], self.commands[e, :2]), -2.0, 2.0))
            r += c.rew_alive
            fell = (base_z < c.term_min_height) or (grav[2] > c.term_grav_z)
            died[e] = fell
            if fell:
                r += c.rew_termination
            team[e] = r
            # per-leg energy + smoothness
            tau = d.actuator_force[self.all_act]
            for i in range(len(LEG_NAMES)):
                sl = slice(i * JOINTS_PER_LEG, (i + 1) * JOINTS_PER_LEG)
                leg_pen[e, i] = (
                    c.pen_torque * np.sum(tau[sl] ** 2)
                    + c.pen_action_rate * np.sum((self.last_action[e, i] - self.prev_action[e, i]) ** 2)
                )
        # feet air-time gait reward (team-level): reward swing duration at touchdown,
        # gated by a non-trivial command -> pushes the policy off the standing optimum.
        contact_b = contacts > 0.5
        first_contact = contact_b & (~self.last_contact)
        self.feet_air_time += self.dt
        cmd_active = (
            (np.linalg.norm(self.commands[:, :2], axis=1) > c.cmd_active_thresh)
            | (np.abs(self.commands[:, 2]) > c.cmd_active_thresh)
        ).astype(np.float32)
        air_reward = c.rew_feet_air_time * np.sum(
            (self.feet_air_time - c.feet_air_time_target) * first_contact, axis=1
        ) * cmd_active
        self.feet_air_time *= (~contact_b)
        self.last_contact = contact_b
        team = team + air_reward.astype(np.float32)

        self._died = died
        per_agent = team[:, None] + leg_pen
        timeout = self.step_buf >= self.max_steps
        done = died | timeout
        info = {"died": died, "timeout": timeout}
        return per_agent.astype(np.float32), done, np.zeros(self.n, dtype=bool), info

    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray):
        """actions [N,4,3] in (-1,1) -> (exo[N,4,22], state[N,52], reward[N,4], done[N], bootstrap_obs).

        On done, the env auto-resets and the returned obs is the NEW episode's;
        the terminal state for bootstrapping (time-limit truncation) is returned
        separately as ``term_state`` so the trainer can value-bootstrap.
        """
        self.prev_action[:] = self.last_action
        self.last_action[:] = np.asarray(actions, dtype=np.float32).reshape(
            self.n, len(LEG_NAMES), JOINTS_PER_LEG)
        self._apply_pd_and_step(self.last_action)
        self.step_buf += 1

        reward, done, success, info = self._rewards_dones()
        # terminal state (pre-reset) for truncation bootstrap
        _, term_state = self._observations()
        # auto-reset done envs
        done_idx = np.nonzero(done)[0]
        if len(done_idx):
            # resample command for survivors crossing the command-hold window
            self._reset_idx(done_idx)
        # also resample commands periodically for alive envs
        live = np.nonzero(~done & (self.step_buf % self.cmd_resample == 0))[0]
        if len(live):
            self._resample_cmd(live)

        exo, state = self._observations()
        info["term_state"] = term_state
        # truncation = timeout (not a true terminal); death = true terminal
        info["truncated"] = info["timeout"] & ~info["died"]
        return exo, state, reward, done, info
