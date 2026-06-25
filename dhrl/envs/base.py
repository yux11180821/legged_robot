# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-agent env + simulation-backend interfaces.

Problem (§3): MDP <S, A, T, R> with N agents in M species; homogeneous species
share one policy & reward (Eq.1).  We split the two interfaces so the cooperative
TASK logic (goals, reward, observation assembly) is decoupled from the SIMULATOR
(Habitat / IsaacSim / a mock for tests):

  * ``Backend``        -- drives the simulator: apply per-agent low-level actions,
                          report agent positions / headings / proprioception.
  * ``MultiAgentEnv``  -- a task built on a backend: returns per-agent (exo, proprio),
                          a shared reward and an episode-done flag each step.

Each ``step`` returns, PER AGENT:
    exo     [N, exo_dim]      -> the trainable UL+ML actor's input  (e_t)
    proprio [N, proprio_dim]  -> the frozen LL's input              (p_t)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Abstract simulator wrapper that the cooperative task envs drive each step.

    Why: the paper's METHOD (the 3-layer HRL policy + IPPO) is independent of which
    physics simulator produces the robot state.  By funnelling every simulator-specific
    operation through this small interface, the task envs (Corridor Crossing, etc.) and
    the frozen Lower-Layer locomotion operator stay simulator-agnostic: the paper used
    IsaacSim + Ant robots, our reproduction uses Habitat-sim + Spot (see
    ``habitat_world.HabitatBackend``), and unit tests use a lightweight mock.  Only the
    concrete subclass knows about the actual engine.

    Concrete subclasses must expose ``n_agents`` and implement the abstract methods that
    read out the simulated robot state needed to assemble the exteroceptive observation
    ``e_t`` (Upper-Layer input) and the proprioceptive observation ``p_t`` (frozen
    Lower-Layer input).

    Attributes:
        n_agents: number of homogeneous agents N in the simulated scene.
    """

    n_agents: int

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulator to a fresh episode (re-spawn all N agents/objects).

        Why: called by ``MultiAgentEnv.reset`` at the start of every episode so the task
        env can re-sample goals from the new agent layout.

        Returns:
            None -- state is read back afterwards via the accessor methods.
        """
        ...

    @abstractmethod
    def step(self, actions: np.ndarray) -> None:
        """Apply per-agent low-level actions and advance the simulator one control step.

        Why: these are the FROZEN Lower-Layer operator's outputs a_t (paper §4.3,
        Fig.2), i.e. the raw locomotion commands the simulator's controller consumes;
        the task env never interprets them, it only forwards them here.

        Args:
            actions: [N, action_dim] float -- the action a_t for each of the N agents
                produced by the Lower Layer for this control step.

        Returns:
            None -- the resulting state is read back via the accessor methods.
        """

    @abstractmethod
    def agent_positions(self) -> np.ndarray:
        """Return the ground-plane position of every agent.

        Why: positions feed both the reward (progress toward goals) and the
        exteroceptive observation e_t (relative goal + nearest-neighbour vectors, §4.2).

        Returns:
            [N, 2] float -- (x, y) ground position of each agent in world coordinates.
        """

    @abstractmethod
    def agent_headings(self) -> np.ndarray:
        """Return the yaw heading of every agent.

        Why: headings rotate world-frame relative vectors into each agent's EGOCENTRIC
        frame when building e_t (the partial observation of §4.2/Fig.3).

        Returns:
            [N] float -- heading of each agent in radians.
        """

    @abstractmethod
    def proprio(self) -> np.ndarray:
        """Return the proprioceptive observation of every agent.

        Why: this is p_t, the FROZEN Lower-Layer operator's input (paper §4.3, Fig.2) --
        the agent's own body state (local position/velocity/torque), distinct from the
        exteroceptive e_t that the trainable Upper Layer consumes.

        Returns:
            [N, proprio_dim] float -- per-agent proprioceptive vector.
        """

    def collisions(self) -> int:
        """Return the cumulative inter-agent collision count since reset.

        Why: the cooperative reward subtracts a penalty for collisions (e.g. Corridor
        Crossing, Eq.1); the task env differences this counter to get the per-step delta.
        Default 0 lets simulators that cannot report collisions opt out harmlessly.

        Returns:
            int -- running total of inter-agent collisions in the current episode.
        """
        return 0

    def sample_navigable(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one random navigable ground point (e.g. for goal/spawn placement).

        Why: tasks that place goals on free floor (rather than the swap layout) draw
        reachable targets from the simulator's navigation mesh.

        Args:
            rng: NumPy Generator providing reproducible randomness for the draw.

        Returns:
            [2] float -- (x, y) ground coordinate of a navigable point.

        Raises:
            NotImplementedError: if the concrete backend has no navigation mesh.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release simulator resources (windows, GPU contexts, file handles).

        Why: optional cleanup hook; the default no-op suits backends (e.g. the mock)
        that hold nothing to release.

        Returns:
            None.
        """
        pass


class MultiAgentEnv(ABC):
    """Abstract cooperative task: the per-agent MDP <S, A, T, R> the IPPO runner trains on.

    Why: this is the stage-2 environment of the two-stage recipe (§5.2).  A concrete task
    (Cooperative Transport / Corridor Crossing / Ravine Bridging, §5.1/Fig.1) owns the
    goals, the SHARED homogeneous reward (Eq.1), and the assembly of the two observation
    streams; it delegates all physics to a ``Backend``.  The runner instantiates E copies
    and stacks their outputs into an [E, N] batch of parallel environments, then optimises
    the Upper+Middle layers with Independent PPO (each agent on its OWN observation, no
    centralized critic, no global-state concatenation -- the decentralization of §4.2/§5.4).

    Per step the env emits, for every agent, the two distinct observation streams of the
    3-layer policy (Fig.2): the exteroceptive ``exo`` (e_t -> trainable Upper Layer) and
    the proprioceptive ``proprio`` (p_t -> frozen Lower Layer).

    Attributes:
        n_agents: number of homogeneous agents N (parameters and reward are shared, Eq.1).
        exo_dim: width of the exteroceptive vector e_t fed to the Upper Layer.
        proprio_dim: width of the proprioceptive vector p_t fed to the frozen Lower Layer.
        command_dim: width of the command the Middle Layer emits to the Lower Layer
            (this command is the PPO action variable -- UL+ML are trained, LL is frozen).
        action_dim: width of the low-level action a_t the Lower Layer sends to the sim.
    """

    n_agents: int
    exo_dim: int
    proprio_dim: int
    command_dim: int
    action_dim: int

    @abstractmethod
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Start a fresh episode and return the initial observations for all agents.

        Why: re-spawns the scene (via the backend) and re-samples goals, then hands the
        runner the first e_t / p_t so the HRL policy can produce its opening action.

        Returns:
            tuple (exo, proprio):
                exo     [N, exo_dim]     -- exteroceptive obs e_t (Upper-Layer input).
                proprio [N, proprio_dim] -- proprioceptive obs p_t (frozen Lower-Layer
                                            input).
        """

    @abstractmethod
    def step(self, actions: np.ndarray):
        """Advance the task one control step and return the transition for all agents.

        Why: applies the frozen Lower-Layer actions through the backend, recomputes the
        SHARED reward (Eq.1) and the episode-termination logic, and re-reads the next
        observations.  The ``done`` / ``info`` contract is what lets the runner do a
        truncation-aware GAE: on a TIME-LIMIT truncation (not a true terminal) the value
        function is bootstrapped from the terminal observation supplied in ``info``.

        Args:
            actions: [N, action_dim] float -- low-level action a_t per agent emitted by
                the frozen Lower Layer for this step.

        Returns:
            tuple (exo, proprio, reward, done, info):
                exo     [N, exo_dim]     -- next exteroceptive obs e_t.
                proprio [N, proprio_dim] -- next proprioceptive obs p_t.
                reward  [N] float        -- SHARED reward broadcast to every agent (Eq.1).
                done    bool             -- True on task success (a TRUE terminal) or on
                                            hitting the time limit (a TRUNCATION).
                info    dict             -- may carry ``truncated`` (bool: time-limit vs
                                            true terminal) and ``term_exo`` (the terminal
                                            e_t used to bootstrap V in truncation-aware GAE).
        """

    def close(self) -> None:
        """Release the task's resources by closing the underlying backend.

        Why: optional cleanup hook so the runner can tear down E parallel envs cleanly;
        the default no-op suits tasks with nothing to release.

        Returns:
            None.
        """
        pass
