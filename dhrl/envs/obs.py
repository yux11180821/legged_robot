# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Observation construction (paper §4.2, Fig.3).

  proprioceptive p_t : local position / velocity / joint torque -- the LOWER layer only.
  exteroceptive  e_t : [ environmental obs , NEAREST-NEIGHBOUR relative obs ].

The 'nearest-neighbour only' partial observation is the paper's scalability design:
each agent sees just its CLOSEST peer, never the global state.  That is what makes
the method scale to many agents and transfer zero-shot (§5.4) -- and what we must NOT
violate by concatenating everyone's obs.

Pure numpy; unit-testable.
"""
from __future__ import annotations

import numpy as np


def world_to_ego(d: np.ndarray, heading: float) -> tuple[float, float]:
    """World ground displacement ``d=[dx, dy]`` -> agent-ego (forward, lateral),
    with the agent's heading along +forward."""
    c, s = np.cos(heading), np.sin(heading)
    return float(d[0] * c + d[1] * s), float(-d[0] * s + d[1] * c)


def nearest_neighbour(positions: np.ndarray, i: int) -> int:
    """Index of the closest OTHER agent to agent ``i``.  positions: ``[N, 2]``."""
    d = np.linalg.norm(positions - positions[i], axis=-1)
    d[i] = np.inf
    return int(np.argmin(d))


def relative_ego(positions: np.ndarray, headings: np.ndarray, i: int, target: np.ndarray):
    """``target`` (world ``[2]``) relative to agent ``i`` -> ``[forward, lateral, distance]``."""
    d = np.asarray(target, dtype=np.float32) - positions[i]
    fwd, lat = world_to_ego(d, headings[i])
    return np.array([fwd, lat, float(np.hypot(d[0], d[1]))], dtype=np.float32)


def nearest_neighbour_ego(positions: np.ndarray, headings: np.ndarray, i: int) -> np.ndarray:
    """Agent ``i``'s nearest neighbour in ``i``'s ego frame -> ``[forward, lateral, distance]``."""
    j = nearest_neighbour(positions, i)
    return relative_ego(positions, headings, i, positions[j])


def build_exteroceptive(positions: np.ndarray, headings: np.ndarray,
                        goals: np.ndarray | None = None,
                        env_obs: np.ndarray | None = None,
                        dist_norm: float = 10.0) -> np.ndarray:
    """Per-agent e_t = [ goal-ego(3) (if a goal task) , nearest-neighbour-ego(3) ,
    env_obs(optional) ], all distances /dist_norm and clipped to [-1, 1].

    Returns ``[N, exo_dim]``.  Crucially this is built PER AGENT from only that agent's
    nearest neighbour -- no agent ever sees the full set of positions.
    """
    n = len(positions)
    rows = []
    for i in range(n):
        parts = []
        if goals is not None:
            parts.append(relative_ego(positions, headings, i, goals[i]) / dist_norm)
        parts.append(nearest_neighbour_ego(positions, headings, i) / dist_norm)
        if env_obs is not None:
            parts.append(np.asarray(env_obs[i], dtype=np.float32))
        rows.append(np.clip(np.concatenate(parts), -1.0, 1.0))
    return np.stack(rows).astype(np.float32)
