from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class DHRLConfig:
    scenario: str
    num_agents: int = 4
    num_envs: int = 64
    seed: int = 0
    total_steps: int = 200_000
    rollout_steps: int = 64
    max_episode_steps: int = 300
    proprio_dim: int = 8
    extero_dim: int = 12
    command_dim: int = 2
    feature_dim: int = 64
    hidden_dim: int = 64
    lr: float = 5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 4096
    command_limit: float = 1.0
    curriculum_steps: int = 100_000
    log_interval: int = 10
    eval_interval: int = 25_000
    eval_episodes: int = 50
    output_dir: str = "results/dhrl_proxy"
    lower_layer_mode: str = "velocity"
    task: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> DHRLConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return DHRLConfig(**data)


def save_config(config: DHRLConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, sort_keys=True)
