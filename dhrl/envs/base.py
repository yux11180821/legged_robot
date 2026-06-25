# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Base classes for the cooperation tasks and the simulation backend.

Plain duck-typed classes (no ABCs).  We keep the TASK (goals, reward, observation
assembly) separate from the SIMULATOR so the same task runs on Habitat or on a mock:

  * ``Backend``       -- drives a simulator: apply actions, read positions / headings /
                         proprioception.  Subclasses (HabitatBackend, a test mock) just
                         fill in the methods below.
  * ``MultiAgentEnv`` -- a cooperation task built on a backend; each ``step`` returns,
                         per agent, the exteroceptive obs ``exo`` (e_t, Upper-Layer input)
                         and the proprioceptive obs ``proprio`` (p_t, frozen Lower-Layer
                         input), a shared reward (Eq.1), a done flag and an info dict.
"""
import numpy as np


class Backend:
    """Simulator wrapper the tasks drive.  Subclasses set ``n_agents`` and fill these in."""

    n_agents: int

    def reset(self):
        """Reset the simulator to a fresh episode."""
        raise NotImplementedError

    def step(self, actions):
        """Apply per-agent low-level actions ``[N, action_dim]`` for one control step."""
        raise NotImplementedError

    def agent_positions(self):
        """Ground positions ``[N, 2]`` of all agents."""
        raise NotImplementedError

    def agent_headings(self):
        """Headings ``[N]`` (radians) of all agents."""
        raise NotImplementedError

    def proprio(self):
        """Proprioceptive obs ``[N, proprio_dim]`` (the frozen Lower Layer's input p_t)."""
        raise NotImplementedError

    def collisions(self):
        """Cumulative inter-agent collision count (0 if the simulator can't report it)."""
        return 0

    def sample_navigable(self, rng):
        """A random navigable ground point ``[2]`` (for goal placement)."""
        raise NotImplementedError

    def close(self):
        """Release simulator resources (default no-op)."""
        pass


class MultiAgentEnv:
    """A cooperation task (§5.1).  Subclasses set the dims below and implement reset/step.

    The runner builds E copies and stacks them into an [E, N] batch for IPPO.  Subclasses
    set: ``n_agents``, ``exo_dim`` (e_t width), ``proprio_dim`` (p_t width),
    ``command_dim`` (the ML command = the PPO action), ``action_dim`` (the LL's a_t).
    """

    n_agents: int
    exo_dim: int
    proprio_dim: int
    command_dim: int
    action_dim: int

    def get_env_info(self):
        """Return the dict the learner needs to size its networks (MARL-style env_info)."""
        return {
            "n_agents": self.n_agents,
            "exo_dim": self.exo_dim,
            "proprio_dim": self.proprio_dim,
            "command_dim": self.command_dim,
            "action_dim": self.action_dim,
        }

    def reset(self):
        """Start a fresh episode -> (exo [N, exo_dim], proprio [N, proprio_dim])."""
        raise NotImplementedError

    def step(self, actions):
        """Apply ``actions`` [N, action_dim] -> (exo, proprio, reward [N], done, info).

        ``done`` is True on success (true terminal) or time limit (truncation, flagged by
        ``info['truncated']``); on truncation ``info['term_exo']`` carries the terminal e_t
        used to bootstrap V in truncation-aware GAE.
        """
        raise NotImplementedError

    def close(self):
        """Release the task's resources (default no-op)."""
        pass
