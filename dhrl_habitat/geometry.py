# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Ground-plane geometry shared by every stage (single source of truth).

Verified conventions (habitat-lab source, see selftest.py for the worked cases):

  * World frame is Y-up; the ground plane is (x, z).
  * ``LocalizationSensor`` returns ``[x, y, z, heading]`` where ``heading`` is the
    signed angle of the agent's local +X (forward) axis, computed by
    ``get_angle_to_pos`` (tasks/rearrange/utils.py): for a forward axis (fx, fz)
    in world,  heading = -atan2(fz, fx).
    Equivalently the forward axis in world is  f = (cos(h), -sin(h))  in (x, z).
  * ``BaseVelNonCylinderAction.step`` maps the 3-D action (longitudinal,
    lateral, angular) to LOCAL velocities ``mn.Vector3(longitudinal, 0,
    -lateral)`` and ``angular = (0, ang, 0)`` (actions.py:725-728).  So:
      - command vx (+1) moves along local +X (forward = world (cos h, -sin h));
      - command vy (+1) moves along local -Z (world (-sin h, -cos h));
      - command wz (+1) is rotation about +Y, which INCREASES heading h.

  Therefore the EGO frame used everywhere here is:
      ego_forward axis = local +X,   ego_lateral axis = local -Z,
  and a positive turn (wz > 0) rotates the forward axis toward +lateral?  No --
  check: at h=0 forward=(1,0), lateral=-z=(0,-1); after +dh, forward=(cos dh,
  -sin dh) which moves toward -z = +lateral.  Yes: wz>0 turns toward +lateral,
  so the turn needed to face a target at ego (f, l) is  delta = atan2(l, f).

Pure numpy; unit-tested in selftest.py with hand-checked cases.
"""

from __future__ import annotations

import numpy as np


def world_to_ego(dx: float | np.ndarray, dz: float | np.ndarray, heading: float | np.ndarray):
    """World ground-plane displacement (dx, dz) -> (ego_forward, ego_lateral).

    ego_forward = d . f       with f = (cos h, -sin h)
    ego_lateral = d . (-z_l)  with local z axis in world = (sin h, cos h)
    """
    ch, sh = np.cos(heading), np.sin(heading)
    ego_f = dx * ch - dz * sh
    ego_l = -dx * sh - dz * ch
    return ego_f, ego_l


def heading_delta_to(ego_f: float | np.ndarray, ego_l: float | np.ndarray):
    """Signed turn (radians) that faces the target at ego (f, l); wz>0 == +delta."""
    return np.arctan2(ego_l, ego_f)


def wrap_angle(a: float | np.ndarray):
    """Wrap to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def ground_pos(localization: np.ndarray) -> np.ndarray:
    """LocalizationSensor [x, y, z, heading] -> ground (x, z)."""
    loc = np.asarray(localization, dtype=np.float32).reshape(-1)
    return np.array([loc[0], loc[2]], dtype=np.float32)


def heading_of(localization: np.ndarray) -> float:
    loc = np.asarray(localization, dtype=np.float32).reshape(-1)
    return float(loc[3])


def rel_target_ego(localization: np.ndarray, target_xz: np.ndarray):
    """-> (ego_forward, ego_lateral, euclidean_dist) of target in the agent frame."""
    p = ground_pos(localization)
    h = heading_of(localization)
    d = np.asarray(target_xz, dtype=np.float32) - p
    ego_f, ego_l = world_to_ego(float(d[0]), float(d[1]), h)
    return float(ego_f), float(ego_l), float(np.hypot(d[0], d[1]))


# --------------------------------------------------------------------------- #
# Manual command (人工指令) -- the advisor's validation baseline.
# --------------------------------------------------------------------------- #
def manual_velocity_command(ego_f: float, ego_l: float, dist: float,
                            *, turn_gain: float = 1.5, stop_dist: float = 0.3) -> np.ndarray:
    """Hand-crafted holonomic command toward a target at ego (f, l).

    Returns (vx, vy, wz) in [-1, 1]^3: drive along the ego direction of the
    target (slowing within 1 m), turn to face it.  Deterministic, instant.
    """
    if dist < stop_dist:
        return np.zeros(3, dtype=np.float32)
    scale = min(1.0, dist) / max(dist, 1e-6)
    vx = ego_f * scale
    vy = ego_l * scale
    wz = float(np.clip(turn_gain * heading_delta_to(ego_f, ego_l), -1.0, 1.0))
    return np.clip(np.array([vx, vy, wz], dtype=np.float32), -1.0, 1.0)


def manual_position_command(ego_f: float, ego_l: float, dist: float,
                            *, radius: float = 2.0, stop_dist: float = 0.3) -> np.ndarray:
    """Position-mode command: ego waypoint clipped to ``radius``, normalized to [-1,1].

    Third component = normalized turn-to-face (delta / pi).
    """
    if dist < stop_dist:
        return np.zeros(3, dtype=np.float32)
    clip = min(1.0, radius / max(dist, 1e-6))
    cx = ego_f * clip / radius
    cy = ego_l * clip / radius
    cw = float(heading_delta_to(ego_f, ego_l)) / np.pi
    return np.clip(np.array([cx, cy, cw], dtype=np.float32), -1.0, 1.0)


def build_command(mode: str, ego_f: float, ego_l: float, dist: float, **kw) -> np.ndarray:
    if mode == "velocity":
        return manual_velocity_command(ego_f, ego_l, dist, **kw)
    if mode == "position":
        return manual_position_command(ego_f, ego_l, dist, **kw)
    raise ValueError(f"unknown lower-control mode: {mode}")
