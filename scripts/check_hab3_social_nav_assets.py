#!/usr/bin/env python3
"""Preflight check for Habitat-3 social-navigation assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from unitree_env import (  # noqa: E402
    missing_social_nav_assets,
    social_nav_download_command,
    unresolved_social_nav_lfs_pointers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        default=str(PROJECT_ROOT),
        help="Habitat-Lab project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    missing = missing_social_nav_assets(project_dir)
    unresolved = unresolved_social_nav_lfs_pointers(project_dir)
    if not missing and not unresolved:
        print("Habitat-3 social-nav assets: OK")
        return

    print("Habitat-3 social-nav assets are incomplete:", file=sys.stderr)
    if missing:
        print("", file=sys.stderr)
        print("Missing files/directories:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
    if unresolved:
        print("", file=sys.stderr)
        print("Unresolved Git LFS pointer files:", file=sys.stderr)
        for path in unresolved:
            print(f"  - {path}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Download command:", file=sys.stderr)
    print(f"  cd {project_dir}", file=sys.stderr)
    print("  source /root/miniconda3/etc/profile.d/conda.sh", file=sys.stderr)
    print("  conda activate habitat", file=sys.stderr)
    print(f"  {social_nav_download_command()}", file=sys.stderr)
    print("", file=sys.stderr)
    print("If HuggingFace is blocked, configure a mirror first:", file=sys.stderr)
    print(
        '  git config --global url."https://hf-mirror.com/".insteadOf '
        '"https://huggingface.co/"',
        file=sys.stderr,
    )
    print(
        "If the mirror still fails on cas-bridge.xethub, copy the resolved "
        "Habitat-3 data directory from another machine or dataset cache.",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
