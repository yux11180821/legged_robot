from __future__ import annotations

from pathlib import Path
import csv
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .buffer import MultiAgentRolloutBuffer
from .config import DHRLConfig, save_config
from .models import make_policy
from .proxy_envs import make_env


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def evaluate(policy: nn.Module, config: DHRLConfig, device: torch.device, episodes: int) -> dict[str, float]:
    eval_config = DHRLConfig(**{**config.__dict__, "num_envs": min(episodes, config.num_envs), "seed": config.seed + 10_000})
    env = make_env(eval_config)
    obs = env.reset()
    hidden = policy.initial_hidden((env.num_envs, env.num_agents), device)  # type: ignore[attr-defined]
    successes: list[float] = []
    returns = np.zeros((env.num_envs, env.num_agents), dtype=np.float32)
    while len(successes) < episodes:
        with torch.no_grad():
            out = policy.act(to_tensor(obs.proprio, device), to_tensor(obs.extero, device), hidden, deterministic=True)  # type: ignore[attr-defined]
        obs, reward, done, info = env.step(out.action.cpu().numpy())
        hidden = out.hidden * (1.0 - to_tensor(done, device)[..., None])
        returns += reward
        done_env = info["episode_done"].astype(bool)
        for idx in np.where(done_env)[0]:
            successes.append(float(info["success"][idx]))
            if len(successes) >= episodes:
                break
        returns[done_env] = 0.0
    return {"success_rate": float(np.mean(successes)), "episodes": float(len(successes))}


def train(config: DHRLConfig, variant: str, device_name: str = "auto") -> dict[str, Any]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    set_seed(config.seed)
    env = make_env(config)
    policy = make_policy(
        variant,
        proprio_dim=config.proprio_dim,
        extero_dim=config.extero_dim,
        command_dim=config.command_dim,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        command_limit=config.command_limit,
        lower_layer_mode=config.lower_layer_mode,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    out_dir = Path(config.output_dir) / config.scenario / variant / f"seed_{config.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.json")
    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "episode_reward", "success_rate", "eval_success_rate", "variant", "seed"])
        writer.writeheader()

    obs = env.reset()
    hidden = policy.initial_hidden((env.num_envs, env.num_agents), device)  # type: ignore[attr-defined]
    global_step = 0
    episode_returns = np.zeros((env.num_envs, env.num_agents), dtype=np.float32)
    recent_rewards: list[float] = []
    recent_success: list[float] = []

    while global_step < config.total_steps:
        buffer = MultiAgentRolloutBuffer(config.rollout_steps, env.num_envs, env.num_agents, config.proprio_dim, config.extero_dim, config.command_dim, config.hidden_dim, device)
        for _ in range(config.rollout_steps):
            proprio = to_tensor(obs.proprio, device)
            extero = to_tensor(obs.extero, device)
            with torch.no_grad():
                out = policy.act(proprio, extero, hidden)  # type: ignore[attr-defined]
            next_obs, reward_np, done_np, info = env.step(out.action.cpu().numpy())
            reward = to_tensor(reward_np, device)
            done = to_tensor(done_np, device)
            buffer.add(proprio, extero, out.action.detach(), out.log_prob.detach(), reward, done, out.value.detach(), hidden.detach())
            hidden = out.hidden.detach() * (1.0 - done[..., None])
            obs = next_obs
            global_step += env.num_envs * env.num_agents
            episode_returns += reward_np
            done_env = info["episode_done"].astype(bool)
            if done_env.any():
                recent_rewards.extend(episode_returns[done_env].mean(axis=1).tolist())
                recent_success.extend(info["success"][done_env].tolist())
                episode_returns[done_env] = 0.0

        with torch.no_grad():
            next_out = policy.act(to_tensor(obs.proprio, device), to_tensor(obs.extero, device), hidden, deterministic=True)  # type: ignore[attr-defined]
        buffer.compute_returns(next_out.value.detach(), config.gamma, config.gae_lambda)

        for _ in range(config.update_epochs):
            for mb in buffer.minibatches(config.minibatch_size):
                new = policy.evaluate_actions(mb["proprio"], mb["extero"], mb["hiddens"], mb["actions"])  # type: ignore[attr-defined]
                ratio = (new.log_prob - mb["log_probs"]).exp()
                unclipped = ratio * mb["advantages"]
                clipped = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * mb["advantages"]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (new.value - mb["returns"]).pow(2).mean()
                entropy = new.entropy.mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        if global_step % (config.log_interval * env.num_envs * env.num_agents * config.rollout_steps) < env.num_envs * env.num_agents * config.rollout_steps:
            eval_success = ""
            if config.eval_interval > 0 and global_step % config.eval_interval < env.num_envs * env.num_agents * config.rollout_steps:
                eval_success = evaluate(policy, config, device, config.eval_episodes)["success_rate"]
                torch.save({"model": policy.state_dict(), "config": config.__dict__, "variant": variant, "step": global_step}, out_dir / f"ckpt_{global_step}.pt")
            row = {
                "step": global_step,
                "episode_reward": float(np.mean(recent_rewards[-100:])) if recent_rewards else 0.0,
                "success_rate": float(np.mean(recent_success[-100:])) if recent_success else 0.0,
                "eval_success_rate": eval_success,
                "variant": variant,
                "seed": config.seed,
            }
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
            print(row, flush=True)

    torch.save({"model": policy.state_dict(), "config": config.__dict__, "variant": variant, "step": global_step}, out_dir / "final.pt")
    return {"output_dir": str(out_dir), "step": global_step}
