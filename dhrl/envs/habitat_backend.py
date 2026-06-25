# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Habitat-sim ``Backend`` (our reproduction platform).

The paper runs in IsaacSim with Ant robots; we reproduce the METHOD in Habitat-sim
(ReplicaCAD, Spot).  This wraps a multi-agent Habitat gym env behind the ``Backend``
interface so the task envs stay simulator-agnostic.  Habitat is imported lazily, so
this module imports fine on a machine without Habitat -- it only needs Habitat when a
``HabitatBackend`` is actually constructed (i.e. on the training server).

Ported from the old ``dhrl_habitat/habitat_io.py`` (verified there): the wrapper chain
``GymHabitatEnv.env.env._env._sim``, the LocalizationSensor convention
(``[x, y, z, heading]``; forward = (cos h, -sin h) in the (x, z) plane), and the
NavMesh point sampling.
"""
from __future__ import annotations

import numpy as np

from .base import Backend

_ACTION_KEY = "agent_{}_base_velocity"
_LOC_KEY = "agent_{}_localization_sensor"


class HabitatBackend(Backend):
    """Concrete ``Backend`` backed by a multi-agent Habitat-sim environment.

    Why: this is our reproduction platform.  The paper trains in IsaacSim with Ant robots;
    we reproduce the METHOD in Habitat-sim (ReplicaCAD scenes, Spot robots) by wrapping a
    multi-agent Habitat gym env behind the simulator-agnostic ``Backend`` interface, so the
    cooperative task envs and the frozen Lower-Layer operator need no Habitat knowledge.
    Habitat is imported LAZILY (only inside ``_build``), so this module imports cleanly on a
    machine without Habitat installed -- the dependency is needed only when a backend is
    actually constructed (i.e. on the training server).  The wrapper chain, the
    LocalizationSensor convention ([x, y, z, heading]; forward = (cos h, -sin h) in the
    (x, z) plane), and the NavMesh sampling are ported from the verified legacy
    ``dhrl_habitat/habitat_io.py``.
    """

    def __init__(self, *, config_name: str, seed: int, n_agents: int,
                 max_episode_steps: int = 500, action_dim: int = 3,
                 enable_lateral: bool = True, strip_render: bool = True):
        """Build and bind a seeded multi-agent Habitat env and cache its observation keys.

        Why: precomputes the per-agent action / localization-sensor dictionary keys (Habitat
        namespaces observations as ``agent_{i}_...``) and reaches down to the underlying
        ``RearrangeSim`` so navigation-mesh sampling works.  ``proprio_dim`` is 4 because the
        per-agent proprioception we expose is the LocalizationSensor reading [x, y, z, heading].

        Args:
            config_name: str -- Habitat config name resolving to the scene/agents/task setup.
            seed: int -- RNG seed applied to both ``habitat.seed`` and ``simulator.seed``.
            n_agents: int -- number of agents N in the scene (sets up the per-agent keys).
            max_episode_steps: int -- Habitat episode horizon (time limit -> truncation).
            action_dim: int -- width of each agent's low-level base-velocity action a_t.
            enable_lateral: bool -- if True, turn on lateral (sideways) base motion in actions.
            strip_render: bool -- if True, drop per-step GPU sensors (obs is localization-only).

        Returns:
            None.
        """
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.proprio_dim = 4  # localization [x, y, z, heading] per agent
        self._action_keys = [_ACTION_KEY.format(i) for i in range(n_agents)]  # "agent_i_base_velocity"
        self._loc_keys = [_LOC_KEY.format(i) for i in range(n_agents)]        # "agent_i_localization_sensor"
        self._env = self._build(config_name, seed, max_episode_steps, enable_lateral, strip_render)
        self._sim = self._get_sim(self._env)  # unwrap to RearrangeSim for the pathfinder
        self._obs: dict | None = None         # latest observation dict, populated by reset/step

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build(config_name, seed, max_episode_steps, enable_lateral, strip_render):
        """Construct the seeded Habitat gym env, applying lateral-move and render overrides.

        Why: centralises the LAZY Habitat import and the config edits the reproduction needs
        -- seeding for determinism, enabling sideways base motion so agents can yield in the
        corridor, and (optionally) stripping all rendering sensors so each step is pure
        localization with NO per-step GPU render, which is far faster for our
        localization-only observation pipeline.

        Args:
            config_name: str -- Habitat config name to load.
            seed: int -- seed injected into both habitat.seed and simulator.seed.
            max_episode_steps: int -- per-episode step limit override.
            enable_lateral: bool -- enable lateral base movement on any action that supports it.
            strip_render: bool -- if True, clear each agent's ``sim_sensors`` (no rendering).

        Returns:
            A constructed Habitat gym env (``GymHabitatEnv``) ready to reset/step.
        """
        import habitat
        from habitat.config import read_write
        from habitat.gym import make_gym_from_config

        cfg = habitat.get_config(config_name, overrides=[
            f"habitat.seed={seed}", f"habitat.simulator.seed={seed}",
            f"habitat.environment.max_episode_steps={max_episode_steps}",
        ])
        with read_write(cfg):
            if enable_lateral:
                for key, ac in cfg.habitat.task.actions.items():  # scan every action spec
                    if hasattr(ac, "enable_lateral_move"):        # only those that support sideways motion
                        ac.enable_lateral_move = True
            if strip_render:  # obs is localization-only -> no per-step GPU render
                for ag in list(cfg.habitat.simulator.agents.keys()):
                    cfg.habitat.simulator.agents[ag].sim_sensors = {}  # remove cameras/depth/etc.
        return make_gym_from_config(cfg)

    @staticmethod
    def _get_sim(gym_env):
        """Unwrap the gym wrapper chain to reach the underlying ``RearrangeSim`` simulator.

        Why: ``sample_navigable`` needs the raw simulator's pathfinder, but Habitat buries
        it under several gym ``.env`` wrappers; this peels up to six layers off and then
        probes the known attribute names (``habitat_env`` / ``_env`` then ``sim`` / ``_sim``)
        following the verified legacy access path.

        Args:
            gym_env: the constructed Habitat gym env returned by ``_build``.

        Returns:
            The inner ``RearrangeSim`` instance (exposes ``.pathfinder``).

        Raises:
            AttributeError: if the simulator cannot be reached through the wrapper chain.
        """
        e = gym_env
        for _ in range(6):                # peel up to 6 gym wrapper layers
            if hasattr(e, "env"):
                e = e.env
            else:
                break
        inner = getattr(e, "habitat_env", None) or getattr(e, "_env", None)  # the Habitat Env object
        sim = getattr(inner, "sim", None) or getattr(inner, "_sim", None)    # its simulator handle
        if sim is None:
            raise AttributeError("could not reach RearrangeSim through the gym wrapper chain")
        return sim

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the Habitat episode and cache the initial observation dict.

        Why: implements ``Backend.reset``; the cached ``self._obs`` is what the accessor
        methods (positions/headings/proprio) read from until the next step.

        Returns:
            None.
        """
        self._obs = self._env.reset()

    def step(self, actions: np.ndarray) -> None:
        """Flatten the per-agent actions and advance the Habitat env one control step.

        Why: implements ``Backend.step``; Habitat's gym env expects all agents' base-velocity
        actions concatenated into a single flat vector, so we reshape [N, action_dim] -> [N*action_dim].
        The returned observation and info dicts are cached for the accessors / ``collisions``.

        Args:
            actions: [N, action_dim] float -- frozen Lower-Layer base-velocity action a_t per agent.

        Returns:
            None -- updates cached ``self._obs`` and ``self._info``.
        """
        flat = np.asarray(actions, dtype=np.float32).reshape(-1)  # concat all agents into one flat action vector
        self._obs, _, _, self._info = self._env.step(flat)        # reward/done from Habitat are ignored; the task env computes its own

    def _loc(self) -> np.ndarray:
        """Stack every agent's LocalizationSensor reading from the cached observation dict.

        Why: a single helper so positions/headings/proprio all derive from the same
        [x, y, z, heading] convention; per-agent readings live under the ``agent_{i}_localization_sensor``
        keys precomputed in ``__init__``.

        Returns:
            [N, 4] float -- per-agent localization [x, y, z, heading].
        """
        return np.stack([np.asarray(self._obs[k], dtype=np.float32) for k in self._loc_keys])

    def agent_positions(self) -> np.ndarray:
        """Return the ground-plane (x, z) position of every agent.

        Why: implements ``Backend.agent_positions`` used for reward/progress and for building
        e_t (§4.2).  Habitat's up-axis is y, so the ground plane is spanned by columns (x, z)
        of the localization vector.

        Returns:
            [N, 2] float -- (x, z) ground position per agent.
        """
        loc = self._loc()
        return loc[:, [0, 2]]  # ground plane (x, z); column 1 (y) is the up-axis, dropped

    def agent_headings(self) -> np.ndarray:
        """Return the yaw heading of every agent.

        Why: implements ``Backend.agent_headings`` -- the heading (column 3 of the
        localization reading) rotates relative vectors into each agent's egocentric frame
        when assembling e_t.

        Returns:
            [N] float -- heading per agent in radians.
        """
        return self._loc()[:, 3]

    def proprio(self) -> np.ndarray:
        """Return the per-agent proprioceptive observation p_t (the full localization vector).

        Why: implements ``Backend.proprio``; in this reproduction p_t is the raw
        LocalizationSensor reading [x, y, z, heading], i.e. the frozen Lower-Layer operator's
        body-state input (§4.3).

        Returns:
            [N, 4] float -- per-agent [x, y, z, heading].
        """
        return self._loc()  # [N, 4] = [x, y, z, heading]

    def collisions(self) -> int:
        """Return the inter-agent collision count reported by the last Habitat step.

        Why: implements ``Backend.collisions``; the task reward subtracts a penalty for fresh
        collisions (Eq.1).  Reads the ``num_agents_collide`` measure from the cached info dict,
        defaulting to 0 if the measure is absent or the info dict is not yet populated.

        Returns:
            int -- number of agent-agent collisions from the most recent step (0 if unavailable).
        """
        try:
            return int(self._info.get("num_agents_collide", 0))
        except (AttributeError, TypeError, ValueError):  # info missing / wrong type / non-int value
            return 0

    def sample_navigable(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one random navigable ground point from the scene's navigation mesh.

        Why: implements ``Backend.sample_navigable`` for tasks that place goals on free floor.
        Prefers the largest indoor island (so points land inside the usable room, not in a
        disconnected mesh fragment), and retries up to 20 times to reject non-finite draws.
        Note: the ``rng`` argument matches the interface, but Habitat's pathfinder samples with
        its OWN (config-seeded) RNG, so determinism comes from the env seed rather than ``rng``.

        Args:
            rng: NumPy Generator (interface-compatibility; not used by Habitat's pathfinder).

        Returns:
            [2] float -- (x, z) ground coordinate of a navigable point.

        Raises:
            RuntimeError: if 20 attempts all yield non-finite points.
        """
        island = getattr(self._sim, "_largest_indoor_island_idx", None)  # restrict to main room if available
        pf = self._sim.pathfinder
        for _ in range(20):                                              # retry to reject non-finite samples
            p = (pf.get_random_navigable_point(island_index=island) if island is not None
                 else pf.get_random_navigable_point())
            p = np.asarray(p, dtype=np.float32)
            if np.isfinite(p).all():
                return np.array([p[0], p[2]], dtype=np.float32)          # drop the y (up) component
        raise RuntimeError("pathfinder failed to sample a navigable point")

    def close(self) -> None:
        """Close the underlying Habitat env, swallowing any teardown error.

        Why: implements ``Backend.close``; Habitat teardown can raise during interpreter
        shutdown, and a failure to release the sim is non-fatal for the run, so the exception
        is intentionally suppressed.

        Returns:
            None.
        """
        try:
            self._env.close()
        except Exception:  # teardown errors are non-fatal; ignore them
            pass
