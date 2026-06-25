# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Task -- Corridor Crossing (paper Fig.1b, homogeneous).

Agents must pass through a narrow corridor; only one can pass at a time, so
"yield / concession" behaviour has to emerge.  We realise it with the swap layout:
each agent's goal is another agent's start, forcing them to cross each other.

Reward is SHARED across agents (Eq.1, homogeneous):
    sum_i progress_i (toward own goal)  + first-arrival bonus per agent
    + all-arrived bonus                 - inter-agent collision penalty  - slack
Episode ends on all-arrived (true terminal) or time limit (truncation -> the runner
bootstraps V(term_exo); ``info['truncated']`` / ``info['term_exo']``).

The task is decoupled from the simulator via a ``Backend`` (so it is testable with a
mock); positions/headings/proprio/collisions come from the backend.
"""
from __future__ import annotations

import numpy as np

from . import obs as O
from .base import Backend, MultiAgentEnv


class CorridorCrossing(MultiAgentEnv):
    """Corridor Crossing cooperative task (Fig.1b, homogeneous), a concrete ``MultiAgentEnv``.

    Why: a narrow corridor admits only one agent at a time, so the agents must learn
    "yield / concession" behaviour to avoid deadlock -- a coordination skill the paper's
    decentralized HRL policy is meant to discover.  We force the conflict with a SWAP
    layout: agent i's goal is agent (i+1)'s start position, so every agent must traverse
    the bottleneck and cross the others.  The reward is SHARED across the homogeneous
    agents (Eq.1): team progress toward goals, per-agent first-arrival bonuses, an
    all-arrived bonus, minus a collision penalty and a small per-step slack.  Episode ends
    on all-arrived (a TRUE terminal) or the time limit (a TRUNCATION, which the runner
    treats with a value bootstrap from ``info['term_exo']``).
    """

    def __init__(self, backend: Backend, *, proprio_dim: int, command_dim: int,
                 action_dim: int, max_steps: int = 500, success_dist: float = 0.5,
                 dist_norm: float = 10.0, progress_w: float = 1.0, arrival_bonus: float = 2.5,
                 all_arrived_bonus: float = 10.0, collision_pen: float = 0.25,
                 slack: float = 0.01, rng: np.random.Generator | None = None):
        """Configure the task: bind a simulator backend and the reward/termination knobs.

        Why: the dimensions are wired so the task can hand the right-shaped tensors to the
        3-layer policy, and the reward coefficients realise the SHARED cooperative reward
        of Eq.1.  ``exo_dim`` is fixed to 6 because the Upper-Layer observation e_t is the
        PARTIAL observation of §4.2/Fig.3: an egocentric 3-vector to the agent's own goal
        plus an egocentric 3-vector to ONLY the nearest neighbour (the scalability design),
        not to all peers.

        Args:
            backend: ``Backend`` -- simulator wrapper providing positions/headings/proprio.
            proprio_dim: int -- width of p_t fed to the frozen Lower Layer.
            command_dim: int -- width of the Middle-Layer command (the PPO action variable).
            action_dim: int -- width of the low-level action a_t sent to the simulator.
            max_steps: int -- episode time limit; reaching it is a TRUNCATION not a terminal.
            success_dist: float -- L2 radius (m) within which an agent counts as "arrived".
            dist_norm: float -- divisor that normalises relative vectors when building e_t.
            progress_w: float -- weight on team distance-reduction progress in the reward.
            arrival_bonus: float -- one-off reward each time a new agent first arrives.
            all_arrived_bonus: float -- terminal bonus when every agent has arrived.
            collision_pen: float -- penalty per new inter-agent collision this step.
            slack: float -- small constant subtracted every step to discourage dawdling.
            rng: optional NumPy Generator for reproducible sampling (default: fresh one).

        Returns:
            None.
        """
        self.backend = backend
        self.n_agents = backend.n_agents
        self.proprio_dim = proprio_dim
        self.command_dim = command_dim
        self.action_dim = action_dim
        self.exo_dim = 6  # goal-ego(3) + nearest-neighbour-ego(3)
        self.max_steps = max_steps
        self.success_dist = success_dist
        self.dist_norm = dist_norm
        self.progress_w = progress_w
        self.arrival_bonus = arrival_bonus
        self.all_arrived_bonus = all_arrived_bonus
        self.collision_pen = collision_pen
        self.slack = slack
        self.rng = rng or np.random.default_rng()
        self._goals = None

    def _exo(self) -> np.ndarray:
        """Assemble the exteroceptive observation e_t for every agent.

        Why: e_t is the Upper-Layer input (Fig.2) and, per §4.2/Fig.3, is a PARTIAL
        observation -- egocentric goal vector plus the egocentric vector to the NEAREST
        neighbour only.  Delegates the egocentric rotation and nearest-neighbour selection
        to the shared ``partial_observation`` builder so every task assembles e_t identically.

        Returns:
            [N, exo_dim] float -- the partial exteroceptive observation e_t per agent.
        """
        return O.build_exteroceptive(self.backend.agent_positions(), self.backend.agent_headings(),
                                     goals=self._goals, dist_norm=self.dist_norm)

    def _dists(self) -> np.ndarray:
        """Compute each agent's current straight-line distance to its own goal.

        Why: the per-step reduction in this distance is the team-progress reward term, and
        crossing below ``success_dist`` flags an arrival.

        Returns:
            [N] float -- L2 distance (m) from each agent to its assigned goal.
        """
        return np.linalg.norm(self.backend.agent_positions() - self._goals, axis=-1)

    def _sample_goals(self) -> None:
        """Assign goals with the swap layout that forces the corridor conflict.

        Why: setting goal_i to the START position of agent (i+1) mod N guarantees every
        agent must traverse the bottleneck and cross the others, which is what makes
        "yield" behaviour necessary (Fig.1b).  ``np.roll(..., shift=-1)`` over the position
        array performs exactly this cyclic shift of start positions into goals.

        Returns:
            None -- stores the [N, 2] goal array in ``self._goals``.
        """
        # swap: goal_i = start of agent (i+1) % N -> forced crossing
        self._goals = np.roll(self.backend.agent_positions().copy(), shift=-1, axis=0)

    def reset(self):
        """Begin a new episode: respawn the scene, assign goals, clear progress trackers.

        Why: implements ``MultiAgentEnv.reset`` -- reseeds the per-episode bookkeeping
        (previous distances for the progress term, arrival flags, collision baseline, step
        counter) and returns the first observation pair so the policy can act.

        Returns:
            tuple (exo, proprio):
                exo     [N, exo_dim]     -- initial partial exteroceptive obs e_t.
                proprio [N, proprio_dim] -- initial proprioceptive obs p_t (frozen LL input).
        """
        self.backend.reset()
        self._sample_goals()
        self._prev_dist = self._dists()                  # baseline for the progress term
        self._arrived = np.zeros(self.n_agents, dtype=bool)  # per-agent first-arrival latch
        self._prev_coll = 0                              # collision counter baseline (for delta)
        self._t = 0                                      # step counter (vs max_steps for truncation)
        return self._exo(), self.backend.proprio()

    def step(self, actions):
        """Advance one control step and return the SHARED-reward transition for all agents.

        Why: implements ``MultiAgentEnv.step`` for Corridor Crossing.  It forwards the
        frozen Lower-Layer actions to the simulator, then composes the cooperative reward
        of Eq.1 from four homogeneous, team-level terms (progress + arrival bonuses
        - collisions - slack) and broadcasts that single scalar to every agent (so all
        agents optimise the same objective).  It also distinguishes a TRUE terminal
        (all-arrived) from a TIME-LIMIT TRUNCATION, exporting ``term_exo`` for the runner's
        truncation-aware GAE bootstrap.

        Args:
            actions: [N, action_dim] float -- low-level action a_t per agent from the LL.

        Returns:
            tuple (exo, proprio, reward, done, info):
                exo     [N, exo_dim]     -- next partial exteroceptive obs e_t.
                proprio [N, proprio_dim] -- next proprioceptive obs p_t.
                reward  [N] float        -- the shared scalar reward broadcast to all agents.
                done    bool             -- True on all-arrived (terminal) or time limit.
                info    dict             -- ``truncated`` (time-limit vs terminal),
                                            ``success`` (all-arrived), ``arrived`` (count),
                                            and ``term_exo`` (terminal e_t for GAE bootstrap).
        """
        self.backend.step(np.asarray(actions, dtype=np.float32))
        self._t += 1

        cur = self._dists()
        progress = float(np.sum(self._prev_dist - cur))      # team progress (shared); sum of per-agent distance reductions
        self._prev_dist = cur                                # roll baseline forward for next step

        newly = (cur < self.success_dist) & (~self._arrived) # agents arriving for the FIRST time this step
        self._arrived |= cur < self.success_dist             # latch arrivals (stay arrived even if they drift out)
        all_arrived = bool(self._arrived.all())

        coll = self.backend.collisions()
        coll_delta = max(0, coll - self._prev_coll)          # new collisions since last step (clamp >=0)
        self._prev_coll = coll

        shared = (self.progress_w * progress                 # +team progress toward goals
                  + self.arrival_bonus * float(np.sum(newly))  # +one-off bonus per newly-arrived agent
                  + (self.all_arrived_bonus if all_arrived else 0.0)  # +terminal team bonus
                  - self.collision_pen * coll_delta          # -penalty for fresh collisions
                  - self.slack)                              # -constant per-step pressure to finish
        reward = np.full(self.n_agents, shared, dtype=np.float32)  # homogeneous shared reward (Eq.1): same scalar to every agent

        truncated = (self._t >= self.max_steps) and not all_arrived  # time-limit cutoff that is NOT a true success
        done = all_arrived or self._t >= self.max_steps
        exo = self._exo()
        info = {"truncated": truncated, "success": all_arrived,
                "arrived": int(self._arrived.sum()), "term_exo": exo}  # term_exo: bootstrap V here on truncation
        return exo, self.backend.proprio(), reward, done, info

    def close(self):
        """Tear down the task by closing the underlying simulator backend.

        Why: implements ``MultiAgentEnv.close`` so the runner can release the E parallel
        envs' simulator resources cleanly.

        Returns:
            None.
        """
        self.backend.close()
