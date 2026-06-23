# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Lower-layer per-leg quadruped MARL package.

Importing this package registers the gym task id so ``gym.make(...)`` works:

    Isaac-Quadruped-FourLegs-Direct-MARL-v0

Train it with MAPPO (centralized critic) or IPPO (decentralized critic).
"""

import os

import gymnasium as gym

from . import agents
from .quadruped_four_legs_env import QuadrupedFourLegsEnv, QuadrupedFourLegsEnvCfg

_AGENTS_DIR = os.path.dirname(agents.__file__)

gym.register(
    id="Isaac-Quadruped-FourLegs-Direct-MARL-v0",
    entry_point=f"{__name__}.quadruped_four_legs_env:QuadrupedFourLegsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_four_legs_env:QuadrupedFourLegsEnvCfg",
        # skrl multi-agent entry points (consumed by --algorithm MAPPO / IPPO).
        "skrl_mappo_cfg_entry_point": os.path.join(_AGENTS_DIR, "skrl_mappo_cfg.yaml"),
        "skrl_ippo_cfg_entry_point": os.path.join(_AGENTS_DIR, "skrl_ippo_cfg.yaml"),
    },
)

__all__ = ["QuadrupedFourLegsEnv", "QuadrupedFourLegsEnvCfg"]
