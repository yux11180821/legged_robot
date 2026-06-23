"""Habitat-3 social-navigation definitions for the D-HRL reproduction.

The filename is kept for compatibility with earlier scripts.  The content here
is not a Unitree single-agent environment: it describes the two-agent
Habitat-3 Spot + humanoid social-nav setup used by the custom PPO runner in
``train_ddp.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlgorithmSpec:
    label: str
    use_memory: bool
    description: str


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "dhrl": AlgorithmSpec(
        label="Distributed HRL (RNN memory)",
        use_memory=True,
        description=(
            "Custom multi-agent PPO with centralized critic, per-agent action "
            "heads, and recurrent spatiotemporal memory."
        ),
    ),
    "no_memory": AlgorithmSpec(
        label="No Spatiotemporal Memory",
        use_memory=False,
        description=(
            "Same Habitat-3 two-agent PPO setup, with the recurrent memory "
            "removed for the paper's no-memory ablation."
        ),
    ),
}


REQUIRED_SOCIAL_NAV_ASSETS: tuple[str, ...] = (
    "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json",
    "data/scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json",
    "data/scene_datasets/hssd-hab/scene_filter_files/105515448_173104512.rec_filter.json",
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


def resolve_habitat_config_name(config_name: str) -> str:
    """Map the old baselines social-nav config to the Habitat-Lab env config."""
    normalized = config_name.strip()
    if normalized in {
        "social_nav/social_nav",
        "social_nav/social_nav.yaml",
        "habitat-baselines/habitat_baselines/config/social_nav/social_nav.yaml",
    }:
        return "benchmark/multi_agent/hssd_spot_human_social_nav.yaml"
    return normalized


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
