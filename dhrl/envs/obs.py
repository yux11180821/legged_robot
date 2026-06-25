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
    """Rotate a world-frame ground displacement into the agent's egocentric frame.

    The exteroceptive obs e_t is expressed egocentrically so the shared policy is invariant to
    the agent's absolute world heading (a permutation/pose-invariance that helps the homogeneous
    parameter sharing and zero-shot transfer, §4.2).  Applies a 2-D rotation by -heading so the
    agent's facing direction maps to +forward.

    Args:
        d: world-frame ground displacement [dx, dy], shape [2].
        heading (float): the agent's yaw in world frame (radians).

    Returns:
        (forward, lateral) coordinates of ``d`` in the agent's ego frame (Python floats).
    """
    c, s = np.cos(heading), np.sin(heading)
    # Standard 2-D rotation by -heading: forward = d.x*cos+d.y*sin, lateral = -d.x*sin+d.y*cos.
    return float(d[0] * c + d[1] * s), float(-d[0] * s + d[1] * c)


def nearest_neighbour(positions: np.ndarray, i: int) -> int:
    """Find the index of the closest OTHER agent to agent ``i``.

    This single nearest-neighbour selection is the heart of the paper's partial-observation /
    scalability design (§4.2, Fig.3): each agent attends to only its closest peer rather than the
    full set, so the obs dimension is constant regardless of the agent count.

    Args:
        positions: ground positions of all agents, shape [N, 2].
        i (int): index of the query agent.

    Returns:
        int index j != i of the nearest other agent.
    """
    d = np.linalg.norm(positions - positions[i], axis=-1)  # Euclidean distance to every agent
    d[i] = np.inf  # exclude self so the agent never selects itself as its neighbour
    return int(np.argmin(d))


def relative_ego(positions: np.ndarray, headings: np.ndarray, i: int, target: np.ndarray):
    """Express a world-frame target point relative to agent ``i`` as an ego 3-vector.

    The generic egocentric encoding reused for both the task goal and the nearest-neighbour pose
    that make up e_t (§4.2): it gives the policy a heading-invariant (forward, lateral, range)
    description of any point of interest.

    Args:
        positions: ground positions of all agents, shape [N, 2].
        headings: yaw of all agents in world frame, shape [N].
        i (int): index of the observing agent.
        target: world-frame point of interest [x, y], shape [2].

    Returns:
        np.float32 array [forward, lateral, distance] of ``target`` in agent ``i``'s ego frame,
        shape [3].
    """
    d = np.asarray(target, dtype=np.float32) - positions[i]  # world displacement to the target
    fwd, lat = world_to_ego(d, headings[i])                  # rotate into the agent's ego frame
    # Append the (unrotated) Euclidean range as the third channel.
    return np.array([fwd, lat, float(np.hypot(d[0], d[1]))], dtype=np.float32)


def nearest_neighbour_ego(positions: np.ndarray, headings: np.ndarray, i: int) -> np.ndarray:
    """Compute the ego-frame relative pose of agent ``i``'s nearest neighbour.

    Composes ``nearest_neighbour`` + ``relative_ego`` to produce exactly the inter-agent term of
    e_t (Fig.3): the single closest peer's (forward, lateral, range), the partial observation that
    is the paper's scalability mechanism (§4.2) -- no agent ever encodes the full swarm.

    Args:
        positions: ground positions of all agents, shape [N, 2].
        headings: yaw of all agents in world frame, shape [N].
        i (int): index of the observing agent.

    Returns:
        np.float32 array [forward, lateral, distance] of the nearest neighbour in ``i``'s ego
        frame, shape [3].
    """
    j = nearest_neighbour(positions, i)                    # pick the single closest peer
    return relative_ego(positions, headings, i, positions[j])  # encode its pose egocentrically


def build_exteroceptive(positions: np.ndarray, headings: np.ndarray,
                        goals: np.ndarray | None = None,
                        env_obs: np.ndarray | None = None,
                        dist_norm: float = 10.0) -> np.ndarray:
    """Assemble the full per-agent exteroceptive observation e_t for the UL (§4.2, Fig.3).

    This is the function that materialises the paper's partial observation that the UL perceives:
    for each agent it concatenates the (optional) ego goal, the SINGLE nearest-neighbour ego pose,
    and any environment features -- normalising distances and clipping to a bounded range.  It is
    constructed strictly per agent from that agent's own nearest neighbour, so the obs scales O(1)
    with the swarm and never becomes a global state -- the precise invariant we must not break
    (decentralization, §4.2/§5.4).

    Args:
        positions: ground positions of all agents, shape [N, 2].
        headings: yaw of all agents in world frame, shape [N].
        goals: per-agent world-frame goal points, shape [N, 2], or None for non-goal tasks.
        env_obs: optional per-agent environment features to append, shape [N, env_dim] or None.
        dist_norm (float): distance normaliser dividing the ego (forward, lateral, range) channels.

    Returns:
        np.float32 array of stacked per-agent observations e_t, shape [N, exo_dim], where
        exo_dim = 3*(goals is not None) + 3 + (env feature width if env_obs given), each row
        clipped to [-1, 1].
    """
    n = len(positions)
    rows = []
    for i in range(n):
        parts = []
        if goals is not None:
            # Goal term (only for goal-conditioned tasks): the agent's goal in its ego frame.
            parts.append(relative_ego(positions, headings, i, goals[i]) / dist_norm)
        # Inter-agent term: ONLY the nearest neighbour (the partial-observation scalability design).
        parts.append(nearest_neighbour_ego(positions, headings, i) / dist_norm)
        if env_obs is not None:
            parts.append(np.asarray(env_obs[i], dtype=np.float32))  # optional environment features
        # Concatenate the parts and clip to a bounded range for a well-conditioned policy input.
        rows.append(np.clip(np.concatenate(parts), -1.0, 1.0))
    return np.stack(rows).astype(np.float32)  # [N, exo_dim]
