#!/usr/bin/env python3
# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Author a MINIMAL multi-agent navigation episode set on ReplicaCAD.

RearrangeEmptyTask-v0 only needs the scene; agent starts are sampled at runtime
(min_distance_start_agents) and goals are sampled by the dhrl_habitat adapter.
So each episode is just: scene_id + scene_dataset_config + empty object/target
lists.  Run this ONCE on AutoDL after `datasets_download --uids replica_cad_dataset`:

    python dhrl_habitat/make_replica_cad_episodes.py --episodes-per-scene 50

Writes:
    data/datasets/replica_cad_nav/train/nav_episodes.json.gz
    data/datasets/replica_cad_nav/val/nav_episodes.json.gz
matching the data_path in benchmark/multi_agent/replica_cad_spot_spot.yaml.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from pathlib import Path


def find_scenes(scenes_dir: str) -> list[str]:
    """Return scene_id paths (relative to CWD) for every ReplicaCAD scene."""
    patterns = [
        os.path.join(scenes_dir, "configs", "scenes", "*.scene_instance.json"),
        os.path.join(scenes_dir, "**", "*.scene_instance.json"),
    ]
    found: list[str] = []
    for pat in patterns:
        found = sorted(glob.glob(pat, recursive=True))
        if found:
            break
    # Use paths relative to CWD so habitat resolves them against the run dir.
    return [os.path.relpath(p) for p in found]


def build_episode(ep_id: int, scene_id: str, scene_dataset_config: str) -> dict:
    """Minimal RearrangeEpisode dict (empty objects/targets)."""
    return {
        "episode_id": str(ep_id),
        "scene_id": scene_id,
        "scene_dataset_config": scene_dataset_config,
        "additional_obj_config_paths": [],
        # Placeholder start; multi-agent starts are re-sampled at runtime.
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        # RearrangeSim.reconfigure reads info["object_labels"]; empty for pure nav.
        "info": {"object_labels": {}},
        # RearrangeEpisode required fields, all empty for a pure-nav episode:
        "ao_states": {},
        "rigid_objs": [],
        "targets": {},
        "markers": [],
        "target_receptacles": [],
        "goal_receptacles": [],
        "name_to_receptacle": {},
    }


def validate_with_habitat(path: str) -> bool:
    """Best-effort: confirm habitat can parse the dataset we just wrote."""
    try:
        from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
    except Exception as e:  # habitat not importable here -> skip validation
        print(f"  (skip habitat validation: {type(e).__name__})")
        return True
    ds = RearrangeDatasetV0()
    with gzip.open(path, "rt") as f:
        ds.from_json(f.read())
    print(f"  habitat parsed {len(ds.episodes)} episodes OK; "
          f"ep0 scene_id={ds.episodes[0].scene_id}")
    return len(ds.episodes) > 0


def main() -> None:
    p = argparse.ArgumentParser(description="Author minimal ReplicaCAD nav episodes.")
    p.add_argument("--scenes-dir", default="data/replica_cad")
    p.add_argument("--scene-dataset-config",
                   default="data/replica_cad/replicaCAD.scene_dataset_config.json")
    p.add_argument("--out-root", default="data/datasets/replica_cad_nav")
    p.add_argument("--episodes-per-scene", type=int, default=50)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    if not os.path.exists(args.scene_dataset_config):
        raise SystemExit(
            f"scene dataset config not found: {args.scene_dataset_config}\n"
            "Download it first: python -m habitat_sim.utils.datasets_download "
            "--uids replica_cad_dataset --data-path data --no-replace"
        )

    scenes = find_scenes(args.scenes_dir)
    if not scenes:
        raise SystemExit(f"no *.scene_instance.json under {args.scenes_dir}")
    print(f"found {len(scenes)} ReplicaCAD scenes, e.g. {scenes[:3]}")

    episodes = []
    ep_id = 0
    for scene_id in scenes:
        for _ in range(args.episodes_per_scene):
            episodes.append(build_episode(ep_id, scene_id, args.scene_dataset_config))
            ep_id += 1

    n_val = max(1, int(len(episodes) * args.val_fraction))
    splits = {"val": episodes[:n_val], "train": episodes[n_val:]}
    for split, eps in splits.items():
        out = Path(args.out_root) / split / "nav_episodes.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wt") as f:
            json.dump({"episodes": eps}, f)
        print(f"wrote {len(eps)} episodes -> {out}")
        validate_with_habitat(str(out))

    print("DONE")


if __name__ == "__main__":
    main()
