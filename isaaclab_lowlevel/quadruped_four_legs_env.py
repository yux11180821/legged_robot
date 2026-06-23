# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Lower-layer per-leg MARL locomotion environment (Isaac Lab DirectMARLEnv).

This is the *lower layer* of the hybrid design discussed with the user:

    Upper layer (Commander)  -> trained in Habitat, outputs (vx, vy, wz)         [separate]
    Lower layer (Executors)  -> THIS FILE: one quadruped decomposed into 4 agents

The single robot is split into FOUR cooperating agents, one per leg
(FL / FR / RL / RR).  Each agent:

  * sees ONLY proprioception (its leg's joint state + foot contact + body IMU +
    the velocity command).  No vision, no map  -> matches the "executors cannot
    see the global map / vision-failure" assumption.
  * controls ONLY its own leg's joints (position targets on top of the default
    standing pose, integrated by the robot's PD actuators).

Cooperation = MAPPO with a CENTRALIZED CRITIC (CTDE): the critic reads the full
robot state via ``_get_states()`` while each actor reads only its leg's
observation via ``_get_observations()``.

Reward = SHARED balance/velocity-tracking term (broadcast to all 4 agents,
the "posture-balance shared reward") + a PER-LEG energy/smoothness term (each leg
minimises its own torque/power -> "each leg minimises its own energy").

The velocity command is the INTERFACE to the future upper-layer Commander:
``set_velocity_commands(cmd)`` lets an external policy drive the (frozen) lower
layer.  See ``cfg.external_commands``.

API verified against Isaac Lab v2.x (``isaaclab.*``).  On Isaac Lab <= 1.4.1
replace every ``isaaclab`` prefix with ``omni.isaac.lab`` and ``isaaclab_assets``
with ``omni.isaac.lab_assets``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

# Default robot: Unitree Go2 (12 DoF, 3 per leg).  Verified import path on v2.x.
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# --------------------------------------------------------------------------- #
# Static layout constants.
# --------------------------------------------------------------------------- #
LEG_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")
JOINTS_PER_LEG: int = 3  # Go2: <leg>_{hip,thigh,calf}_joint.  Wheel-legged: bump this.

# Per-agent (one leg) observation layout, all proprioceptive:
#   base_lin_vel_b(3) base_ang_vel_b(3) projected_gravity_b(3) command(3)
#   own_joint_pos_rel(3) own_joint_vel(3) own_foot_force(1) own_last_action(3)
PER_LEG_OBS_DIM: int = 3 + 3 + 3 + 3 + JOINTS_PER_LEG + JOINTS_PER_LEG + 1 + JOINTS_PER_LEG  # = 22

# Centralized critic state (full robot):
#   base_lin_vel_b(3) base_ang_vel_b(3) projected_gravity_b(3) command(3)
#   all_joint_pos_rel(12) all_joint_vel(12) all_foot_forces(4) all_last_actions(12)
NUM_JOINTS: int = JOINTS_PER_LEG * len(LEG_NAMES)  # 12
STATE_DIM: int = 3 + 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + len(LEG_NAMES) + NUM_JOINTS  # = 52


@configclass
class QuadrupedFourLegsEnvCfg(DirectMARLEnvCfg):
    """Config for the per-leg quadruped MARL env."""

    # --- timing ---
    decimation = 4          # 200 Hz sim / 4 -> 50 Hz control
    episode_length_s = 20.0

    # --- MARL spaces (the core DirectMARLEnv contract) ---
    possible_agents = list(LEG_NAMES)
    action_spaces = {leg: JOINTS_PER_LEG for leg in LEG_NAMES}
    observation_spaces = {leg: PER_LEG_OBS_DIM for leg in LEG_NAMES}
    state_space = STATE_DIM  # positive int -> centralized critic via _get_states()

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation)

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # --- robot + foot contact sensor ---
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",  # all bodies (feet + base) in one sensor
        history_length=3,
        update_period=0.0,  # every sim step
        track_air_time=True,
    )

    # Joint / body names.  Indices are resolved at runtime (DOF order is
    # asset-defined, NEVER hardcode -> see __init__).  Swap these for a
    # wheel-legged robot (and add the wheel joint + bump JOINTS_PER_LEG).
    leg_joint_names: dict[str, list[str]] = {
        "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "RL": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        "RR": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
    }
    foot_body_names: dict[str, str] = {
        "FL": "FL_foot", "FR": "FR_foot", "RL": "RL_foot", "RR": "RR_foot",
    }
    base_body_name: str = "base"

    # --- control ---
    action_scale: float = 0.25  # joint target = default_pos + action_scale * action

    # --- observation scales (mild; skrl also running-normalizes) ---
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 0.25
    joint_pos_scale: float = 1.0
    joint_vel_scale: float = 0.05
    contact_force_scale: float = 0.01

    # --- velocity-command sampling (the Commander interface ranges) ---
    external_commands: bool = False  # True -> do not resample; commands come from set_velocity_commands()
    cmd_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    cmd_lin_vel_y: tuple[float, float] = (-0.7, 0.7)
    cmd_ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    # --- SHARED reward scales (posture-balance + velocity tracking) ---
    tracking_sigma: float = 0.25
    rew_track_lin_vel: float = 1.5
    rew_track_ang_vel: float = 0.75
    pen_lin_vel_z: float = -2.0
    pen_ang_vel_xy: float = -0.05
    pen_flat_orientation: float = -2.5
    base_height_target: float = 0.34
    pen_base_height: float = -1.0
    rew_alive: float = 0.5
    rew_termination: float = -50.0  # applied when the robot falls

    # --- PER-LEG reward scales (each leg minimises its own energy/jerk) ---
    pen_torque: float = -2.0e-4       # sum tau^2 over the leg's joints
    pen_power: float = -2.0e-4        # sum |tau * qdot| (mechanical power -> energy)
    pen_joint_accel: float = -2.5e-7
    pen_action_rate: float = -0.01

    # --- termination thresholds ---
    term_base_contact_force: float = 1.0     # N; base touches ground -> fell
    term_tilt_gravity_z: float = -0.5        # projected_gravity_b z > this -> tilted past ~60 deg
    term_min_base_height: float = 0.18       # m


class QuadrupedFourLegsEnv(DirectMARLEnv):
    """One quadruped, four per-leg agents, MAPPO-ready (CTDE)."""

    cfg: QuadrupedFourLegsEnvCfg

    def __init__(self, cfg: QuadrupedFourLegsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve per-leg joint indices at runtime (asset-defined DOF order).
        self._leg_joint_ids: dict[str, torch.Tensor] = {}
        for leg in self.cfg.possible_agents:
            ids, _ = self.robot.find_joints(self.cfg.leg_joint_names[leg])
            self._leg_joint_ids[leg] = torch.as_tensor(ids, dtype=torch.long, device=self.device)

        # Resolve foot + base body indices in the contact sensor.
        foot_ids, _ = self._contact.find_bodies(
            [self.cfg.foot_body_names[leg] for leg in self.cfg.possible_agents]
        )
        self._foot_ids = torch.as_tensor(foot_ids, dtype=torch.long, device=self.device)
        base_ids, _ = self._contact.find_bodies([self.cfg.base_body_name])
        self._base_body_id = int(base_ids[0])

        # Cached default standing joint pose.
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()

        # Persistent buffers (pre-allocated to avoid reference aliasing).
        self._actions = {
            leg: torch.zeros(self.num_envs, JOINTS_PER_LEG, device=self.device)
            for leg in self.cfg.possible_agents
        }
        self._prev_actions = {
            leg: torch.zeros(self.num_envs, JOINTS_PER_LEG, device=self.device)
            for leg in self.cfg.possible_agents
        }
        self._joint_targets = self._default_joint_pos.clone()
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Sanity-check declared dims against the actual layout.
        assert PER_LEG_OBS_DIM == self.cfg.observation_spaces[self.cfg.possible_agents[0]]
        assert STATE_DIM == self.cfg.state_space

    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact = ContactSensor(self.cfg.contact_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact"] = self._contact
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Roll buffers: prev <- current, current <- new.
        targets = self._default_joint_pos.clone()
        for leg in self.cfg.possible_agents:
            self._prev_actions[leg].copy_(self._actions[leg])
            self._actions[leg].copy_(actions[leg])
            ids = self._leg_joint_ids[leg]
            targets[:, ids] = self._default_joint_pos[:, ids] + self.cfg.action_scale * self._actions[leg]
        self._joint_targets = targets

    def _apply_action(self) -> None:
        # Position targets -> the robot's PD actuators do the low-level torque.
        self.robot.set_joint_position_target(self._joint_targets)

    # ------------------------------------------------------------------ #
    # Observations (per-agent, proprioception only) + global state (critic)
    # ------------------------------------------------------------------ #
    def _foot_force_mag(self) -> torch.Tensor:
        """[num_envs, num_legs] contact-force magnitude per foot."""
        forces = self._contact.data.net_forces_w[:, self._foot_ids, :]  # [N, L, 3]
        return torch.norm(forces, dim=-1)

    def _shared_proprio(self) -> torch.Tensor:
        """[num_envs, 12] body terms every agent sees (lin/ang vel, gravity, command)."""
        d = self.robot.data
        return torch.cat(
            [
                d.root_lin_vel_b * self.cfg.lin_vel_scale,
                d.root_ang_vel_b * self.cfg.ang_vel_scale,
                d.projected_gravity_b,
                self._commands
                * torch.tensor(
                    [self.cfg.lin_vel_scale, self.cfg.lin_vel_scale, self.cfg.ang_vel_scale],
                    device=self.device,
                ),
            ],
            dim=-1,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        shared = self._shared_proprio()  # [N, 12]
        joint_pos_rel = (d.joint_pos - self._default_joint_pos) * self.cfg.joint_pos_scale
        joint_vel = d.joint_vel * self.cfg.joint_vel_scale
        foot_force = self._foot_force_mag() * self.cfg.contact_force_scale  # [N, L]

        obs: dict[str, torch.Tensor] = {}
        for i, leg in enumerate(self.cfg.possible_agents):
            ids = self._leg_joint_ids[leg]
            obs[leg] = torch.cat(
                [
                    shared,                              # 12
                    joint_pos_rel[:, ids],               # 3
                    joint_vel[:, ids],                   # 3
                    foot_force[:, i].unsqueeze(-1),      # 1
                    self._actions[leg],                  # 3 (last action)
                ],
                dim=-1,
            )
        return obs

    def _get_states(self) -> torch.Tensor:
        d = self.robot.data
        joint_pos_rel = (d.joint_pos - self._default_joint_pos) * self.cfg.joint_pos_scale
        joint_vel = d.joint_vel * self.cfg.joint_vel_scale
        foot_force = self._foot_force_mag() * self.cfg.contact_force_scale  # [N, L]
        last_actions = torch.cat(
            [self._actions[leg] for leg in self.cfg.possible_agents], dim=-1
        )  # [N, 12]
        return torch.cat(
            [
                d.root_lin_vel_b * self.cfg.lin_vel_scale,    # 3
                d.root_ang_vel_b * self.cfg.ang_vel_scale,    # 3
                d.projected_gravity_b,                        # 3
                self._commands,                               # 3
                joint_pos_rel,                                # 12
                joint_vel,                                    # 12
                foot_force,                                   # 4
                last_actions,                                 # 12
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------ #
    # Reward: shared balance/velocity (broadcast) + per-leg energy
    # ------------------------------------------------------------------ #
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        c = self.cfg

        # --- SHARED (same scalar for all 4 legs) ---
        lin_err = torch.sum((self._commands[:, :2] - d.root_lin_vel_b[:, :2]) ** 2, dim=1)
        track_lin = torch.exp(-lin_err / c.tracking_sigma) * c.rew_track_lin_vel
        ang_err = (self._commands[:, 2] - d.root_ang_vel_b[:, 2]) ** 2
        track_ang = torch.exp(-ang_err / c.tracking_sigma) * c.rew_track_ang_vel

        pen_vz = d.root_lin_vel_b[:, 2] ** 2 * c.pen_lin_vel_z
        pen_wxy = torch.sum(d.root_ang_vel_b[:, :2] ** 2, dim=1) * c.pen_ang_vel_xy
        flat = torch.sum(d.projected_gravity_b[:, :2] ** 2, dim=1) * c.pen_flat_orientation
        height = (d.root_pos_w[:, 2] - c.base_height_target) ** 2 * c.pen_base_height
        alive = torch.full((self.num_envs,), c.rew_alive, device=self.device)
        fell = self._died.float() * c.rew_termination  # _died set in _get_dones (runs first)

        shared = track_lin + track_ang + pen_vz + pen_wxy + flat + height + alive + fell

        # --- PER-LEG (each leg's own energy / smoothness) ---
        torque = d.applied_torque   # [N, num_joints], clamped by the DCMotor model
        jvel = d.joint_vel
        jacc = d.joint_acc
        rewards: dict[str, torch.Tensor] = {}
        for leg in self.cfg.possible_agents:
            ids = self._leg_joint_ids[leg]
            t_pen = torch.sum(torque[:, ids] ** 2, dim=1) * c.pen_torque
            p_pen = torch.sum(torch.abs(torque[:, ids] * jvel[:, ids]), dim=1) * c.pen_power
            a_pen = torch.sum(jacc[:, ids] ** 2, dim=1) * c.pen_joint_accel
            ar_pen = torch.sum((self._actions[leg] - self._prev_actions[leg]) ** 2, dim=1) * c.pen_action_rate
            rewards[leg] = shared + t_pen + p_pen + a_pen + ar_pen
        return rewards

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        d = self.robot.data
        base_force = torch.norm(
            self._contact.data.net_forces_w[:, self._base_body_id, :], dim=-1
        )
        base_contact = base_force > self.cfg.term_base_contact_force
        tilted = d.projected_gravity_b[:, 2] > self.cfg.term_tilt_gravity_z
        too_low = d.root_pos_w[:, 2] < self.cfg.term_min_base_height
        died = base_contact | tilted | too_low
        self._died = died

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {leg: died for leg in self.cfg.possible_agents}
        timeouts = {leg: time_out for leg in self.cfg.possible_agents}
        return terminated, timeouts

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._default_joint_pos[env_ids].clone()
        joint_pos += sample_uniform(-0.1, 0.1, joint_pos.shape, self.device)
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if not self.cfg.external_commands:
            self._resample_commands(env_ids)
        for leg in self.cfg.possible_agents:
            self._actions[leg][env_ids] = 0.0
            self._prev_actions[leg][env_ids] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self._commands[env_ids, 0] = sample_uniform(*self.cfg.cmd_lin_vel_x, (n,), self.device)
        self._commands[env_ids, 1] = sample_uniform(*self.cfg.cmd_lin_vel_y, (n,), self.device)
        self._commands[env_ids, 2] = sample_uniform(*self.cfg.cmd_ang_vel_z, (n,), self.device)

    # ------------------------------------------------------------------ #
    # Upper-layer (Commander) interface
    # ------------------------------------------------------------------ #
    def set_velocity_commands(self, commands: torch.Tensor) -> None:
        """Drive the (frozen) lower layer from an external upper-layer Commander.

        Set ``cfg.external_commands = True`` so resets don't overwrite these.
        ``commands`` is [num_envs, 3] = (vx, vy, wz) in the base frame.
        """
        self._commands[:] = commands.to(self.device)
