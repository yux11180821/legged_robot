#!/usr/bin/env python3
"""Custom Habitat-3 multi-agent PPO runner for the D-HRL reproduction.

This entry point intentionally does not call ``habitat_baselines.run``.  It uses
Habitat-Lab only to build the Habitat-3 social-navigation environment, then runs
our own centralized-critic PPO loop over the two-agent continuous action space.

Algorithms:
  dhrl       : two agent action heads with a GRU spatiotemporal memory.
  no_memory  : same centralized PPO setup without the recurrent memory.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from unitree_env import (
    ALGORITHMS,
    missing_social_nav_assets,
    resolve_habitat_config_name,
    social_nav_download_command,
    unresolved_social_nav_lfs_pointers,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - only used on minimal installs.
    SummaryWriter = None


DEFAULT_ACTION_KEYS = ("agent_0_base_velocity", "agent_1_base_velocity")
METRIC_KEYS = (
    "social_nav_reward",
    "nav_seek_success",
    "dist_to_goal",
    "rot_dist_to_goal",
    "num_steps",
    "num_agents_collide",
    "did_collide",
)


@dataclass(frozen=True)
class RunJob:
    algorithm: str
    seed: int
    gpu: str
    log_file: Path
    cmd: list[str]


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    hidden: torch.Tensor | None
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def parse_ints(raw: Sequence[str]) -> list[int]:
    out: list[int] = []
    for item in raw:
        out.extend(int(x) for x in item.replace(",", " ").split())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run custom Habitat-3 multi-agent D-HRL PPO experiments."
    )
    parser.add_argument(
        "--project-dir",
        default=os.environ.get("PROJECT_DIR", str(Path.cwd())),
        help="Habitat-Lab project root. On AutoDL this is /root/autodl-tmp/habitat-lab.",
    )
    parser.add_argument(
        "--config-name",
        default=os.environ.get(
            "CONFIG_NAME", "benchmark/multi_agent/hssd_spot_human_social_nav.yaml"
        ),
        help=(
            "Habitat-Lab config. The old baselines path social_nav/social_nav.yaml "
            "is accepted and mapped to Habitat-Lab's social-nav benchmark config."
        ),
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
        help="Environment frames per seed.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(os.environ.get("NUM_ENVS", "1")),
        help="Number of Habitat envs stepped sequentially inside each seed worker.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=int(os.environ.get("ROLLOUT_STEPS", "128")),
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=int(os.environ.get("PPO_EPOCHS", "2")),
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=int(os.environ.get("NUM_MINIBATCHES", "2")),
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=int(os.environ.get("HIDDEN_SIZE", "256")),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=int(os.environ.get("IMAGE_SIZE", "32")),
        help="Side length used when compacting image/depth observations.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=int(os.environ.get("MAX_EPISODE_STEPS", "750")),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(os.environ.get("LR", "2.5e-4")),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=float(os.environ.get("GAMMA", "0.99")),
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=float(os.environ.get("GAE_LAMBDA", "0.95")),
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=float(os.environ.get("CLIP_PARAM", "0.2")),
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=float(os.environ.get("VALUE_LOSS_COEF", "0.5")),
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=float(os.environ.get("ENTROPY_COEF", "0.0001")),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=float(os.environ.get("MAX_GRAD_NORM", "0.5")),
    )
    parser.add_argument(
        "--ckpt-interval-frames",
        type=int,
        default=int(os.environ.get("CKPT_INTERVAL_FRAMES", "100000")),
        help="Fixed frame interval for checkpoint saves.",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=int(os.environ.get("NUM_CHECKPOINTS", "0")),
        help="Accepted for compatibility; fixed frame interval controls checkpointing.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.environ.get("LOG_INTERVAL", "10")),
        help="Log every N PPO updates.",
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
        help="Maximum concurrent seed jobs.",
    )
    parser.add_argument(
        "--env-action-keys",
        nargs="+",
        default=list(DEFAULT_ACTION_KEYS),
        help="Habitat action keys controlled by the multi-agent PPO policy.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-asset-check", action="store_true")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", default="0", help=argparse.SUPPRESS)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_assets(args: argparse.Namespace) -> None:
    if args.skip_asset_check:
        return
    config_name = resolve_habitat_config_name(args.config_name)
    if "social_nav" not in config_name and "multi_agent" not in config_name:
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
        + "\n\nDownload them on AutoDL with:\n"
        f"  cd {project_dir}\n"
        "  source /root/miniconda3/etc/profile.d/conda.sh\n"
        "  conda activate habitat\n"
        f"  {social_nav_download_command()}"
    )


def make_jobs(args: argparse.Namespace) -> list[RunJob]:
    project_dir = Path(args.project_dir).resolve()
    seeds = parse_ints(args.seeds)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()] or ["0"]

    jobs: list[RunJob] = []
    for algo_idx, algorithm in enumerate(args.algorithms):
        for seed_idx, seed in enumerate(seeds):
            gpu = gpus[(algo_idx * len(seeds) + seed_idx) % len(gpus)]
            log_file = (
                project_dir
                / "logs"
                / args.exp_name
                / algorithm
                / f"seed_{seed}.log"
            )
            cmd = [
                sys.executable,
                str(project_dir / "train_ddp.py"),
                "--run-one",
                "--project-dir",
                str(project_dir),
                "--config-name",
                args.config_name,
                "--exp-name",
                args.exp_name,
                "--algorithm",
                algorithm,
                "--seed",
                str(seed),
                "--gpu",
                gpu,
                "--total-steps",
                str(args.total_steps),
                "--num-envs",
                str(args.num_envs),
                "--rollout-steps",
                str(args.rollout_steps),
                "--ppo-epochs",
                str(args.ppo_epochs),
                "--num-minibatches",
                str(args.num_minibatches),
                "--hidden-size",
                str(args.hidden_size),
                "--image-size",
                str(args.image_size),
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--lr",
                str(args.lr),
                "--gamma",
                str(args.gamma),
                "--gae-lambda",
                str(args.gae_lambda),
                "--clip-param",
                str(args.clip_param),
                "--value-loss-coef",
                str(args.value_loss_coef),
                "--entropy-coef",
                str(args.entropy_coef),
                "--max-grad-norm",
                str(args.max_grad_norm),
                "--ckpt-interval-frames",
                str(args.ckpt_interval_frames),
                "--log-interval",
                str(args.log_interval),
                "--env-action-keys",
                *args.env_action_keys,
            ]
            if args.skip_asset_check:
                cmd.append("--skip-asset-check")
            jobs.append(RunJob(algorithm, seed, gpu, log_file, cmd))
    return jobs


def launch_job(project_dir: Path, job: RunJob) -> subprocess.Popen:
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
        active.append((job, launch_job(project_dir, job)))

    while active:
        wait_for_slot(active)


def flatten_action_dim(space) -> int:
    from gym import spaces

    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, spaces.Dict) or isinstance(space, Mapping):
        return sum(flatten_action_dim(subspace) for subspace in space.values())
    raise TypeError(f"Unsupported continuous action subspace: {space}")


class ObservationPreprocessor:
    def __init__(self, keys: Sequence[str], image_size: int):
        self.keys = list(keys)
        self.image_size = image_size
        self.feature_dim: int | None = None

    def _compact_image(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            y_idx = np.linspace(0, arr.shape[0] - 1, self.image_size).astype(np.int64)
            x_idx = np.linspace(0, arr.shape[1] - 1, self.image_size).astype(np.int64)
            return arr[np.ix_(y_idx, x_idx)].reshape(-1)
        if arr.ndim == 3:
            y_idx = np.linspace(0, arr.shape[0] - 1, self.image_size).astype(np.int64)
            x_idx = np.linspace(0, arr.shape[1] - 1, self.image_size).astype(np.int64)
            channels = min(arr.shape[2], 3)
            return arr[np.ix_(y_idx, x_idx, np.arange(channels))].reshape(-1)
        return arr.reshape(-1)

    def transform(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in self.keys:
            arr = np.asarray(obs[key], dtype=np.float32)
            if arr.ndim >= 2:
                compact = self._compact_image(arr)
            else:
                compact = arr.reshape(-1)
            compact = np.nan_to_num(compact, nan=0.0, posinf=10.0, neginf=-10.0)
            compact = np.clip(compact, -10.0, 10.0).astype(np.float32, copy=False)
            parts.append(compact)
        out = np.concatenate(parts, dtype=np.float32)
        if self.feature_dim is None:
            self.feature_dim = int(out.shape[0])
        return out


class MultiAgentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        action_splits: Sequence[int],
        *,
        hidden_size: int,
        use_memory: bool,
    ):
        super().__init__()
        self.use_memory = use_memory
        self.hidden_size = hidden_size
        self.action_splits = list(action_splits)
        action_dim = int(sum(action_splits))

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.memory = nn.GRUCell(hidden_size, hidden_size) if use_memory else None
        self.actor_heads = nn.ModuleList(
            nn.Linear(hidden_size, split) for split in self.action_splits
        )
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_memory:
            return None
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def _latent(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = self.encoder(obs)
        if self.memory is None:
            return encoded, None
        if hidden is None:
            hidden = torch.zeros(obs.shape[0], self.hidden_size, device=obs.device)
        next_hidden = self.memory(encoded, hidden)
        return next_hidden, next_hidden

    def _distribution(self, latent: torch.Tensor) -> Normal:
        raw_mean = torch.cat([head(latent) for head in self.actor_heads], dim=-1)
        center = (self.action_high + self.action_low) * 0.5
        scale = (self.action_high - self.action_low) * 0.5
        mean = center + torch.tanh(raw_mean) * scale
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        latent, next_hidden = self._latent(obs, hidden)
        dist = self._distribution(latent)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        env_action = torch.max(torch.min(raw_action, self.action_high), self.action_low)
        return raw_action, env_action, log_prob, value, next_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None,
        raw_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, _ = self._latent(obs, hidden)
        dist = self._distribution(latent)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return log_prob, entropy, value


def make_env(args: argparse.Namespace, env_seed: int):
    import habitat
    from habitat.config import read_write
    from habitat.gym import make_gym_from_config

    config_name = resolve_habitat_config_name(args.config_name)
    overrides = [
        f"habitat.seed={env_seed}",
        f"habitat.simulator.seed={env_seed}",
        f"habitat.environment.max_episode_steps={args.max_episode_steps}",
    ]
    cfg = habitat.get_config(config_name, overrides=overrides)
    with read_write(cfg):
        cfg.habitat.gym.action_keys = list(args.env_action_keys)
    return make_gym_from_config(cfg)


def make_writer(tb_dir: Path):
    if SummaryWriter is None:
        return None
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def save_checkpoint(
    path: Path,
    *,
    model: MultiAgentActorCritic,
    optimizer: torch.optim.Optimizer,
    frames: int,
    args: argparse.Namespace,
    preprocessor: ObservationPreprocessor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frames": frames,
            "algorithm": args.algorithm,
            "seed": args.seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_keys": preprocessor.keys,
            "obs_feature_dim": preprocessor.feature_dim,
            "image_size": preprocessor.image_size,
            "action_keys": list(args.env_action_keys),
        },
        path,
    )


def append_csv(csv_path: Path, row: dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frames", "reward", *METRIC_KEYS, "episode_reward", "episodes"]
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_scalars(writer, row: dict[str, float | int]) -> None:
    if writer is None:
        return
    frames = int(row["frames"])
    writer.add_scalar("reward", float(row["reward"]), frames)
    if "episode_reward" in row:
        writer.add_scalar("episode_reward", float(row["episode_reward"]), frames)
    for key in METRIC_KEYS:
        if key in row:
            value = row[key]
            writer.add_scalar(f"metrics/{key}", float(value), frames)
            if key == "nav_seek_success":
                writer.add_scalar("metrics/social_nav_seek_success", float(value), frames)
    writer.flush()


def build_rollout_batch(
    *,
    obs_buf: list[np.ndarray],
    hidden_buf: list[np.ndarray] | None,
    raw_action_buf: list[np.ndarray],
    old_log_prob_buf: list[np.ndarray],
    reward_buf: list[np.ndarray],
    done_buf: list[np.ndarray],
    value_buf: list[np.ndarray],
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> RolloutBatch:
    rewards = np.asarray(reward_buf, dtype=np.float32)
    dones = np.asarray(done_buf, dtype=np.float32)
    values = np.asarray(value_buf, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_values = next_value
        else:
            next_values = values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values

    obs = torch.as_tensor(np.asarray(obs_buf, dtype=np.float32), device=device).flatten(0, 1)
    raw_actions = torch.as_tensor(
        np.asarray(raw_action_buf, dtype=np.float32), device=device
    ).flatten(0, 1)
    old_log_probs = torch.as_tensor(
        np.asarray(old_log_prob_buf, dtype=np.float32), device=device
    ).flatten(0, 1)
    returns_t = torch.as_tensor(returns, device=device).flatten(0, 1)
    advantages_t = torch.as_tensor(advantages, device=device).flatten(0, 1)
    if hidden_buf is None:
        hidden_t = None
    else:
        hidden_t = torch.as_tensor(
            np.asarray(hidden_buf, dtype=np.float32), device=device
        ).flatten(0, 1)
    advantages_t = (advantages_t - advantages_t.mean()) / (
        advantages_t.std(unbiased=False) + 1e-8
    )
    return RolloutBatch(obs, hidden_t, raw_actions, old_log_probs, returns_t, advantages_t)


def ppo_update(
    model: MultiAgentActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    args: argparse.Namespace,
) -> dict[str, float]:
    batch_size = batch.obs.shape[0]
    minibatch_size = max(1, batch_size // max(1, args.num_minibatches))
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": []}
    for _epoch in range(args.ppo_epochs):
        order = torch.randperm(batch_size, device=batch.obs.device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = order[start : start + minibatch_size]
            mb_hidden = None if batch.hidden is None else batch.hidden[mb_idx]
            new_log_probs, entropy, values = model.evaluate_actions(
                batch.obs[mb_idx],
                mb_hidden,
                batch.raw_actions[mb_idx],
            )
            ratio = torch.exp(new_log_probs - batch.old_log_probs[mb_idx])
            unclipped = ratio * batch.advantages[mb_idx]
            clipped = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
            clipped = clipped * batch.advantages[mb_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (batch.returns[mb_idx] - values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                + args.value_loss_coef * value_loss
                - args.entropy_coef * entropy_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            stats["policy_loss"].append(float(policy_loss.detach().cpu()))
            stats["value_loss"].append(float(value_loss.detach().cpu()))
            stats["entropy"].append(float(entropy_loss.detach().cpu()))
    return {key: float(np.mean(values)) for key, values in stats.items()}


def train_one(args: argparse.Namespace) -> None:
    assert args.algorithm is not None
    assert args.seed is not None
    project_dir = Path(args.project_dir).resolve()
    check_assets(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = [make_env(args, args.seed + idx) for idx in range(max(1, args.num_envs))]
    obs_list = [env.reset() for env in envs]
    obs_keys = list(envs[0].observation_space.spaces.keys())
    preprocessor = ObservationPreprocessor(obs_keys, image_size=args.image_size)
    obs_features = np.stack([preprocessor.transform(obs) for obs in obs_list])

    action_low = np.asarray(envs[0].action_space.low, dtype=np.float32)
    action_high = np.asarray(envs[0].action_space.high, dtype=np.float32)
    action_splits = [
        flatten_action_dim(envs[0].original_action_space.spaces[key])
        for key in args.env_action_keys
    ]
    if sum(action_splits) != int(np.prod(envs[0].action_space.shape)):
        raise RuntimeError(
            f"Action split {action_splits} does not match action space {envs[0].action_space}"
        )

    spec = ALGORITHMS[args.algorithm]
    model = MultiAgentActorCritic(
        obs_dim=int(obs_features.shape[1]),
        action_low=action_low,
        action_high=action_high,
        action_splits=action_splits,
        hidden_size=args.hidden_size,
        use_memory=spec.use_memory,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    hidden = model.initial_hidden(len(envs), device)

    tb_dir = project_dir / "tb" / args.exp_name / args.algorithm / f"seed_{args.seed}"
    ckpt_dir = (
        project_dir
        / "data"
        / "checkpoints"
        / args.exp_name
        / args.algorithm
        / f"seed_{args.seed}"
    )
    csv_path = (
        project_dir
        / "results"
        / args.exp_name
        / f"{args.algorithm}_seed_{args.seed}_scalars.csv"
    )
    writer = make_writer(tb_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    frames = 0
    update_idx = 0
    next_ckpt = max(1, args.ckpt_interval_frames)
    recent_episode_rewards: deque[float] = deque(maxlen=50)
    running_episode_rewards = np.zeros(len(envs), dtype=np.float32)
    last_metrics: dict[str, float] = {}
    last_episode_reward = math.nan

    print(
        f"custom_ppo_start algorithm={args.algorithm} seed={args.seed} "
        f"envs={len(envs)} obs_dim={obs_features.shape[1]} action_splits={action_splits} "
        f"device={device}"
    )

    try:
        while frames < args.total_steps:
            rollout_len = min(
                args.rollout_steps,
                max(1, math.ceil((args.total_steps - frames) / len(envs))),
            )
            obs_buf: list[np.ndarray] = []
            hidden_buf: list[np.ndarray] | None = [] if spec.use_memory else None
            raw_action_buf: list[np.ndarray] = []
            old_log_prob_buf: list[np.ndarray] = []
            reward_buf: list[np.ndarray] = []
            done_buf: list[np.ndarray] = []
            value_buf: list[np.ndarray] = []

            for _step in range(rollout_len):
                obs_tensor = torch.as_tensor(obs_features, dtype=torch.float32, device=device)
                if hidden_buf is not None:
                    assert hidden is not None
                    hidden_buf.append(hidden.detach().cpu().numpy())
                with torch.no_grad():
                    raw_action, env_action, log_prob, value, next_hidden = model.act(
                        obs_tensor, hidden
                    )

                env_action_np = env_action.detach().cpu().numpy().astype(np.float32)
                rewards = np.zeros(len(envs), dtype=np.float32)
                dones = np.zeros(len(envs), dtype=np.float32)
                next_obs_features: list[np.ndarray] = []
                for env_idx, env in enumerate(envs):
                    next_obs, reward, done, info = env.step(env_action_np[env_idx])
                    rewards[env_idx] = float(reward)
                    dones[env_idx] = float(done)
                    running_episode_rewards[env_idx] += float(reward)
                    for key in METRIC_KEYS:
                        if key in info:
                            last_metrics[key] = float(info[key])
                    if done:
                        last_episode_reward = float(running_episode_rewards[env_idx])
                        recent_episode_rewards.append(last_episode_reward)
                        running_episode_rewards[env_idx] = 0.0
                        next_obs = env.reset()
                        if next_hidden is not None:
                            next_hidden[env_idx].zero_()
                    next_obs_features.append(preprocessor.transform(next_obs))

                obs_buf.append(obs_features.copy())
                raw_action_buf.append(raw_action.detach().cpu().numpy())
                old_log_prob_buf.append(log_prob.detach().cpu().numpy())
                reward_buf.append(rewards)
                done_buf.append(dones)
                value_buf.append(value.detach().cpu().numpy())

                obs_features = np.stack(next_obs_features)
                hidden = None if next_hidden is None else next_hidden.detach()
                frames += len(envs)

            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(
                    obs_features, dtype=torch.float32, device=device
                )
                _raw, _env_action, _log_prob, next_value, _next_hidden = model.act(
                    next_obs_tensor, hidden
                )
            batch = build_rollout_batch(
                obs_buf=obs_buf,
                hidden_buf=hidden_buf,
                raw_action_buf=raw_action_buf,
                old_log_prob_buf=old_log_prob_buf,
                reward_buf=reward_buf,
                done_buf=done_buf,
                value_buf=value_buf,
                next_value=next_value.detach().cpu().numpy(),
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                device=device,
            )
            losses = ppo_update(model, optimizer, batch, args)
            update_idx += 1

            if update_idx % max(1, args.log_interval) == 0 or frames >= args.total_steps:
                reward_mean = (
                    float(np.mean(recent_episode_rewards))
                    if recent_episode_rewards
                    else float(np.mean(reward_buf))
                )
                row: dict[str, float | int] = {
                    "frames": frames,
                    "reward": reward_mean,
                    "episodes": len(recent_episode_rewards),
                    **last_metrics,
                }
                if not math.isnan(last_episode_reward):
                    row["episode_reward"] = last_episode_reward
                write_scalars(writer, row)
                append_csv(csv_path, row)
                if writer is not None:
                    for key, value in losses.items():
                        writer.add_scalar(f"losses/{key}", value, frames)
                print(
                    f"frames={frames} reward={reward_mean:.4f} "
                    f"metrics={last_metrics} losses={losses}",
                    flush=True,
                )

            while frames >= next_ckpt:
                save_checkpoint(
                    ckpt_dir / f"ckpt_{next_ckpt:012d}.pt",
                    model=model,
                    optimizer=optimizer,
                    frames=frames,
                    args=args,
                    preprocessor=preprocessor,
                )
                next_ckpt += max(1, args.ckpt_interval_frames)

        save_checkpoint(
            ckpt_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            frames=frames,
            args=args,
            preprocessor=preprocessor,
        )
        print(f"custom_ppo_done frames={frames} final={ckpt_dir / 'final.pt'}")
    finally:
        if writer is not None:
            writer.close()
        for env in envs:
            env.close()


def main() -> None:
    args = parse_args()
    if args.run_one:
        train_one(args)
        return
    check_assets(args)
    jobs = make_jobs(args)
    run_jobs(args, jobs)


if __name__ == "__main__":
    main()
