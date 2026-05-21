"""IsaacSim/Gym integration contract for the D-HRL policy.

The paper trains in IsaacSim/Gym with a pretrained Ant locomotion controller as the lower
layer. This file intentionally avoids importing Isaac-specific modules so that the algorithm can
be tested without the simulator. To connect a real task, implement the following contract:

1. Build per-agent observations:
   - proprio: local locomotion state for the frozen lower layer controller.
   - extero: nearest-neighbour relative state plus scenario features.
2. Call the D-HRL policy to get a high-level command.
3. Feed (proprio, command) to the pretrained lower-layer locomotion policy.
4. Step IsaacSim/Gym with the resulting joint action.

The proxy environments in `proxy_envs.py` consume high-level commands directly, but the same
UL/ML policy and IPPO update are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class IsaacAgentBatch:
    proprio: torch.Tensor
    extero: torch.Tensor
    done: torch.Tensor


class LowerLayerPolicy(Protocol):
    def __call__(self, proprio: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """Return joint-level action for IsaacSim/Gym."""


def dhrl_to_joint_action(policy, lower_layer: LowerLayerPolicy, batch: IsaacAgentBatch, hidden: torch.Tensor, deterministic: bool = False):
    out = policy.act(batch.proprio, batch.extero, hidden, deterministic=deterministic)
    joint_action = lower_layer(batch.proprio, out.action)
    next_hidden = out.hidden * (1.0 - batch.done[..., None])
    return joint_action, next_hidden, out
