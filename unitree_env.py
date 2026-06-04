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


REQUIRED_SOCIAL_NAV_ASSETS: tuple[str, ...] = (
    "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json",
    "data/scene_datasets/hssd-hab/stages/102343992.glb",
    "data/scene_datasets/hssd-hab/objects/0/0001fb06b075a743e6289236cf049df3ad5dfa9c.glb",
    "data/datasets/hssd/rearrange/train/social_rearrange.json.gz",
    "data/humanoids/humanoid_data/female_2/female_2.urdf",
    "data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl",
    "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
    "data/objects/ycb/configs",
    "data/objects/amazon_berkeley/configs",
    "data/objects/google_object_dataset/configs",
)


SOCIAL_NAV_DOWNLOAD_UIDS: tuple[str, ...] = (
    "hssd-hab",
    "hab3-episodes",
    "habitat_humanoids",
    "hab3_bench_assets",
    "hab_spot_arm",
    "ycb",
)


LFS_POINTER_SCAN_ROOTS: tuple[str, ...] = (
    "data/scene_datasets/hssd-hab",
    "data/datasets/hssd",
    "data/humanoids",
    "data/robots/hab_spot_arm",
    "data/objects/ycb",
    "data/objects/amazon_berkeley",
    "data/objects/google_object_dataset",
)

GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def quote_path(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def missing_social_nav_assets(project_dir: Path) -> list[Path]:
    root = Path(project_dir)
    return [
        root / rel_path
        for rel_path in REQUIRED_SOCIAL_NAV_ASSETS
        if not (root / rel_path).exists()
    ]


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as handle:
            return handle.read(len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX
    except OSError:
        return False


def unresolved_social_nav_lfs_pointers(
    project_dir: Path,
    *,
    limit: int = 50,
) -> list[Path]:
    root = Path(project_dir)
    pointers: list[Path] = []
    for rel_root in LFS_POINTER_SCAN_ROOTS:
        scan_root = root / rel_root
        if not scan_root.exists():
            continue
        paths = [scan_root] if scan_root.is_file() else scan_root.rglob("*")
        for path in paths:
            if is_git_lfs_pointer(path):
                pointers.append(path)
                if len(pointers) >= limit:
                    return pointers
    return pointers


def social_nav_download_command() -> str:
    return (
        "python -m habitat_sim.utils.datasets_download --uids "
        + " ".join(SOCIAL_NAV_DOWNLOAD_UIDS)
        + " --data-path data --no-replace"
    )


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
