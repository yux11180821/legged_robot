#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path
from typing import Optional


def load_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def save_json_gz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def parse_scene_tokens(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_scenes(data: dict, scene_tokens: list[str], invert: bool = False) -> dict:
    filtered = dict(data)
    if not scene_tokens:
        filtered["episodes"] = list(data["episodes"])
        return filtered

    def matched(scene_id: str) -> bool:
        hit = any(token in scene_id for token in scene_tokens)
        return not hit if invert else hit

    filtered["episodes"] = [ep for ep in data["episodes"] if matched(ep["scene_id"])]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-scene", default="skokloster-castle")
    parser.add_argument("--test-scene", default="van-gogh-room")
    parser.add_argument(
        "--train-scenes",
        default=None,
        help="Comma-separated scene tokens for train. Overrides --train-scene.",
    )
    parser.add_argument(
        "--test-scenes",
        default=None,
        help="Comma-separated scene tokens for val/test. Overrides --test-scene.",
    )
    parser.add_argument(
        "--train-all-except-test",
        action="store_true",
        help="Use all train episodes whose scene_id does not match test scenes.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)

    train_data = load_json_gz(source_root / "train" / "train.json.gz")
    val_data = load_json_gz(source_root / "val" / "val.json.gz")
    test_data = load_json_gz(source_root / "test" / "test.json.gz")

    train_scenes = parse_scene_tokens(args.train_scenes) or [args.train_scene]
    test_scenes = parse_scene_tokens(args.test_scenes) or [args.test_scene]

    train_filtered = filter_scenes(
        train_data,
        test_scenes if args.train_all_except_test else train_scenes,
        invert=args.train_all_except_test,
    )
    val_filtered = filter_scenes(val_data, test_scenes)
    test_filtered = filter_scenes(test_data, test_scenes)

    if not train_filtered["episodes"]:
        raise RuntimeError(f"No train episodes matched {train_scenes!r}")
    if not val_filtered["episodes"]:
        raise RuntimeError(f"No val episodes matched {test_scenes!r}")

    save_json_gz(out_root / "train" / "train.json.gz", train_filtered)
    save_json_gz(out_root / "val" / "val.json.gz", val_filtered)
    save_json_gz(out_root / "test" / "test.json.gz", test_filtered)

    print(f"train episodes: {len(train_filtered['episodes'])} scenes={train_scenes}")
    print(f"val episodes: {len(val_filtered['episodes'])} scenes={test_scenes}")
    print(f"test episodes: {len(test_filtered['episodes'])} scenes={test_scenes}")
    print(out_root)


if __name__ == "__main__":
    main()
