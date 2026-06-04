"""Multi-agent Habitat experiment definitions for the D-HRL reproduction.

The paper is not a single-agent Unitree navigation problem.  This module keeps
the old filename only for compatibility with earlier scripts, but the content is
now explicitly multi-agent:

* Habitat task/config: social_nav/social_nav.yaml
* Access manager: MultiAgentAccessMgr
* Updater: HRLPPO / HRLDDPPO
* Algorithms: D-HRL with recurrent memory vs no spatiotemporal memory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlgorithmSpec:
    label: str
    rnn_type: str
    description: str
    extra_overrides: tuple[str, ...] = ()


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "dhrl": AlgorithmSpec(
        label="Distributed HRL (RNN memory)",
        rnn_type="LSTM",
        description=(
            "Multi-agent HRL with recurrent spatiotemporal memory. "
            "This is the paper-aligned D-HRL condition."
        ),
    ),
    "no_memory": AlgorithmSpec(
        label="No Spatiotemporal Memory",
        rnn_type="NONE",
        description=(
            "Same multi-agent HRL setup, but the recurrent state encoder is "
            "replaced by an MLP. This matches the paper's no-memory ablation."
        ),
    ),
}


def quote_path(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def build_habitat_overrides(
    *,
    algorithm: str,
    seed: int,
    total_steps: int,
    num_envs: int,
    tensorboard_dir: Path,
    checkpoint_dir: Path,
    num_checkpoints: int,
    log_interval: int,
) -> list[str]:
    """Build Hydra overrides for a multi-agent Habitat-Baselines run."""
    if algorithm not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{algorithm}'. Available: {sorted(ALGORITHMS)}")

    spec = ALGORITHMS[algorithm]
    overrides = [
        f"habitat.seed={seed}",
        f"habitat.simulator.seed={seed}",
        "habitat_baselines.evaluate=False",
        "habitat_baselines.trainer_name=ddppo",
        "habitat_baselines.updater_name=HRLPPO",
        "habitat_baselines.distrib_updater_name=HRLDDPPO",
        "habitat_baselines.rollout_storage_name=HrlRolloutStorage",
        "habitat_baselines.rl.agent.type=MultiAgentAccessMgr",
        "habitat_baselines.rl.agent.num_agent_types=2",
        "habitat_baselines.rl.agent.num_active_agents_per_type=[1,1]",
        "habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1]",
        f"habitat_baselines.rl.ddppo.rnn_type={spec.rnn_type}",
        f"habitat_baselines.total_num_steps={total_steps}",
        f"habitat_baselines.num_environments={num_envs}",
        f"habitat_baselines.num_checkpoints={num_checkpoints}",
        f"habitat_baselines.log_interval={log_interval}",
        f"habitat_baselines.tensorboard_dir={quote_path(tensorboard_dir)}",
        f"habitat_baselines.checkpoint_folder={quote_path(checkpoint_dir)}",
    ]
    overrides.extend(spec.extra_overrides)
    return overrides
