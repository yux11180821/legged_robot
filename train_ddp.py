#!/usr/bin/env python3
"""Launch the multi-agent D-HRL reproduction runs.

This file used to contain a single-agent Unitree/Gym placeholder.  The paper is
about centralized-training/decentralized-execution multi-agent HRL, so the
launcher now targets Habitat's multi-agent social-nav stack instead:

    social_nav/social_nav.yaml
    MultiAgentAccessMgr + HRLPPO

Two algorithms are launched by default:

    dhrl       : recurrent middle layer, LSTM memory
    no_memory  : same multi-agent setup, but rnn_type=NONE

The second one is the paper's "No Spatiotemporal Memory" ablation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from unitree_env import (
    ALGORITHMS,
    build_habitat_overrides,
    missing_social_nav_assets,
    social_nav_download_command,
    unresolved_social_nav_lfs_pointers,
)


@dataclass(frozen=True)
class RunJob:
    algorithm: str
    seed: int
    gpu: str
    log_file: Path
    cmd: list[str]


def parse_ints(raw: Sequence[str]) -> list[int]:
    out: list[int] = []
    for item in raw:
        out.extend(int(x) for x in item.replace(",", " ").split())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Habitat multi-agent D-HRL and no-memory ablation."
    )
    parser.add_argument(
        "--project-dir",
        default=os.environ.get("PROJECT_DIR", str(Path.cwd())),
        help="Habitat-Lab project root. On AutoDL this is /root/autodl-tmp/habitat-lab.",
    )
    parser.add_argument(
        "--config-name",
        default=os.environ.get("CONFIG_NAME", "social_nav/social_nav.yaml"),
        help="Hydra config passed to habitat_baselines.run.",
    )
    parser.add_argument(
        "--exp-name",
        default=os.environ.get("EXP_NAME", "dhrl_social_nav_2algorithms"),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=os.environ.get("ALGORITHMS", "dhrl no_memory").split(),
        choices=sorted(ALGORITHMS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=os.environ.get("SEEDS", "100 200 300 400").split(),
        help="Seed list, e.g. --seeds 100 200 300 400.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=int(float(os.environ.get("TOTAL_STEPS", "2000000"))),
        help="Frames per seed. The paper-scale runs are much longer; 2M is a practical reproduction default.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(os.environ.get("NUM_ENVS", "8")),
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=int(os.environ.get("NUM_CHECKPOINTS", "1")),
    )
    parser.add_argument(
        "--ckpt-interval-frames",
        type=int,
        default=int(os.environ.get("CKPT_INTERVAL_FRAMES", "100000")),
        help="Fixed frame interval used by the patched PPO trainer.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.environ.get("LOG_INTERVAL", "10")),
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("GPUS", "0"),
        help="Comma-separated GPU ids used round-robin.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=int(os.environ.get("MAX_PARALLEL", "1")),
        help="Maximum concurrent seed jobs. Increase carefully on multi-agent scenes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training.",
    )
    parser.add_argument(
        "--skip-asset-check",
        action="store_true",
        help="Skip the Habitat-3 social-nav asset preflight check.",
    )
    return parser.parse_args()


def check_assets(args: argparse.Namespace) -> None:
    if args.skip_asset_check:
        return
    if "social_nav" not in args.config_name:
        return
    project_dir = Path(args.project_dir).resolve()
    missing = missing_social_nav_assets(project_dir)
    unresolved = unresolved_social_nav_lfs_pointers(project_dir)
    if not missing and not unresolved:
        return

    sections: list[str] = []
    if missing:
        sections.append(
            "Missing files/directories:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    if unresolved:
        sections.append(
            "Unresolved Git LFS pointer files:\n"
            + "\n".join(f"  - {path}" for path in unresolved)
        )
    raise SystemExit(
        "Habitat-3 social-nav assets are incomplete:\n"
        + "\n\n".join(sections)
        + "\n\n"
        "Download them on AutoDL with:\n"
        f"  cd {project_dir}\n"
        "  source /root/miniconda3/etc/profile.d/conda.sh\n"
        "  conda activate habitat\n"
        f"  {social_nav_download_command()}\n\n"
        "If HuggingFace is blocked, configure a mirror first:\n"
        "  git config --global url.\"https://hf-mirror.com/\".insteadOf "
        "\"https://huggingface.co/\"\n"
        "If the mirror still fails on cas-bridge.xethub, copy the resolved "
        "Habitat-3 data directory from another machine or dataset cache."
    )


def make_jobs(args: argparse.Namespace) -> list[RunJob]:
    project_dir = Path(args.project_dir).resolve()
    seeds = parse_ints(args.seeds)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        gpus = ["0"]

    jobs: list[RunJob] = []
    for algo_idx, algorithm in enumerate(args.algorithms):
        for seed_idx, seed in enumerate(seeds):
            gpu = gpus[(algo_idx * len(seeds) + seed_idx) % len(gpus)]
            tb_dir = project_dir / "tb" / args.exp_name / algorithm / f"seed_{seed}"
            ckpt_dir = (
                project_dir
                / "data"
                / "checkpoints"
                / args.exp_name
                / algorithm
                / f"seed_{seed}"
            )
            log_file = (
                project_dir
                / "logs"
                / args.exp_name
                / algorithm
                / f"seed_{seed}.log"
            )
            overrides = build_habitat_overrides(
                algorithm=algorithm,
                seed=seed,
                total_steps=args.total_steps,
                num_envs=args.num_envs,
                tensorboard_dir=tb_dir,
                checkpoint_dir=ckpt_dir,
                num_checkpoints=args.num_checkpoints,
                log_interval=args.log_interval,
            )
            cmd = [
                sys.executable,
                "-m",
                "habitat_baselines.run",
                f"--config-name={args.config_name}",
                *overrides,
            ]
            jobs.append(
                RunJob(
                    algorithm=algorithm,
                    seed=seed,
                    gpu=gpu,
                    log_file=log_file,
                    cmd=cmd,
                )
            )
    return jobs


def launch_job(project_dir: Path, job: RunJob, ckpt_interval: int) -> subprocess.Popen:
    job.log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": job.gpu,
            "MAGNUM_LOG": "quiet",
            "HABITAT_SIM_LOG": "quiet",
            "GLOG_minloglevel": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "HYDRA_FULL_ERROR": "1",
            "HABITAT_CKPT_INTERVAL_FRAMES": str(ckpt_interval),
        }
    )
    with job.log_file.open("w") as log:
        log.write(f"algorithm={job.algorithm}\n")
        log.write(f"seed={job.seed}\n")
        log.write(f"gpu={job.gpu}\n")
        log.write("command=" + " ".join(job.cmd) + "\n\n")
    return subprocess.Popen(
        job.cmd,
        cwd=project_dir,
        env=env,
        stdout=job.log_file.open("a"),
        stderr=subprocess.STDOUT,
    )


def wait_for_slot(active: list[tuple[RunJob, subprocess.Popen]]) -> None:
    while True:
        for idx, (job, proc) in enumerate(active):
            status = proc.poll()
            if status is None:
                continue
            active.pop(idx)
            if status != 0:
                raise RuntimeError(
                    f"{job.algorithm} seed {job.seed} failed with code {status}. "
                    f"See {job.log_file}"
                )
            print(f"finished algorithm={job.algorithm} seed={job.seed}")
            return
        time.sleep(2)


def run_jobs(args: argparse.Namespace, jobs: Iterable[RunJob]) -> None:
    project_dir = Path(args.project_dir).resolve()
    active: list[tuple[RunJob, subprocess.Popen]] = []
    for job in jobs:
        spec = ALGORITHMS[job.algorithm]
        print(
            f"launch algorithm={job.algorithm} ({spec.label}) "
            f"seed={job.seed} gpu={job.gpu} log={job.log_file}"
        )
        print(" ".join(job.cmd))
        if args.dry_run:
            continue
        while len(active) >= max(1, args.max_parallel):
            wait_for_slot(active)
        active.append((job, launch_job(project_dir, job, args.ckpt_interval_frames)))

    while active:
        wait_for_slot(active)


def main() -> None:
    args = parse_args()
    check_assets(args)
    jobs = make_jobs(args)
    run_jobs(args, jobs)


if __name__ == "__main__":
    main()
