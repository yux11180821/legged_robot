# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""STAGE 2 runner (paper §5.2.2): with the LL FROZEN, train UL + ML(RNN) with IPPO on a
cooperative task -- fully decentralized.

Call chain (the part the advisor asked to be readable):

    train_upper(cfg)
      └─ build E task envs (HabitatBackend + a task) ─┐
      └─ run_upper(env_fns, cfg) ────────────────────┘
           build  frozen LowerLayer  ─▶ HRLPolicy(UL+ML+frozen LL)  +  DecentralizedCritic
           loop:  IPPO.act ─▶ env.step ─▶ RolloutBuffer.insert
                  RolloutBuffer.compute_returns(truncation-aware GAE)
                  IPPO.update  (clip-PPO; only UL+ML+critic, never the frozen LL)
           save policy + critic

``run_upper`` takes env *factories* so it runs with the real Habitat task OR a mock
backend (tests) without change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dhrl.algorithms.ippo import IPPO
from dhrl.memories.rollout_buffer import RolloutBuffer
from dhrl.networks.critic import DecentralizedCritic
from dhrl.networks.hrl_policy import HRLPolicy
from dhrl.networks.lower_layer import LowerLayer
from dhrl.utils.logging import InferenceTimer, StepLogger


def run_upper(env_fns, cfg: dict, *, device=None, results_dir="dhrl/results", exp_name="upper"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = cfg.get("ppo", {})
    gamma = ppo.get("gamma", 0.995)
    lam = ppo.get("gae_lambda", 0.95)
    T = cfg.get("rollout_steps", 128)
    total = cfg.get("total_steps", 1_000_000)
    H = cfg.get("hidden_size", 256)
    use_memory = cfg.get("use_memory", True)   # False -> 'No Spatiotemporal Memory' ablation

    envs = [fn() for fn in env_fns]
    E, N = len(envs), envs[0].n_agents
    exo_d, pro_d, cmd_d, act_d = (envs[0].exo_dim, envs[0].proprio_dim,
                                  envs[0].command_dim, envs[0].action_dim)

    # frozen lower operator (stage-1 product); fall back to a fresh frozen LL for smoke
    if cfg.get("lower_ckpt") and Path(cfg["lower_ckpt"]).exists():
        lower = LowerLayer.load_frozen(cfg["lower_ckpt"], proprio_dim=pro_d, command_dim=cmd_d,
                                       action_dim=act_d, hidden_size=H)
    else:
        lower = LowerLayer(pro_d, cmd_d, act_d, H).freeze()
    policy = HRLPolicy(exo_d, pro_d, cmd_d, act_d, H, use_memory=use_memory, lower=lower).to(device)
    critic = DecentralizedCritic(exo_d, H).to(device)
    algo = IPPO(policy, critic, lr=ppo.get("lr", 5e-4), clip_param=ppo.get("clip", 0.2),
                entropy_coef=ppo.get("entropy_coef", 1e-3), ppo_epochs=ppo.get("epochs", 4),
                num_minibatches=ppo.get("minibatches", 4), target_kl=cfg.get("target_kl"))
    buffer = RolloutBuffer(T, E, N, exo_d, cmd_d, H, device, use_memory=use_memory)
    logger = StepLogger(Path(results_dir) / f"{exp_name}_seed_{cfg.get('seed', 0)}_metrics.csv",
                        log_every=cfg.get("log_every", 2000),
                        fields=["env_steps", "reward", "success_rate", "infer_ms_median",
                                "policy_loss", "value_loss", "entropy"])
    timer = InferenceTimer(device=str(device))

    def to_t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    resets = [env.reset() for env in envs]
    exo = np.stack([r[0] for r in resets]); proprio = np.stack([r[1] for r in resets])
    hidden = policy.initial_hidden(E * N, device).view(E, N, H) if use_memory else None
    frames, recent_succ = 0, []

    while frames < total:
        buffer.reset()
        for _ in range(T):
            with timer.measure():
                out = algo.act(to_t(exo), to_t(proprio), hidden)
            actions = out["env_action"].cpu().numpy()

            next_exo, next_pro = np.zeros_like(exo), np.zeros_like(proprio)
            reward = np.zeros((E, N), np.float32); done = np.zeros(E, np.float32)
            boot = np.zeros((E, N), np.float32)
            for e, env in enumerate(envs):
                ne, npro, r, d, info = env.step(actions[e])
                reward[e] = r
                if d:
                    done[e] = 1.0
                    if info.get("truncated"):
                        boot[e] = algo.value(to_t(info["term_exo"])[None])[0].cpu().numpy()
                    else:
                        recent_succ.append(1.0 if info.get("success") else 0.0)
                    ne, npro = env.reset()
                next_exo[e], next_pro[e] = ne, npro

            buffer.insert(exo=to_t(exo), hidden=hidden if use_memory else None,
                          raw_command=out["raw_command"], log_prob=out["log_prob"],
                          value=out["value"], reward=to_t(reward), done=to_t(done),
                          bootstrap=to_t(boot))
            exo, proprio = next_exo, next_pro
            hidden = out["next_hidden"]
            if hidden is not None:                       # new episode -> fresh memory
                for e in range(E):
                    if done[e] > 0.5:
                        hidden[e].zero_()
            frames += E
            if logger.should_log(frames):
                logger.log(frames, {"reward": float(reward.mean()),
                                    "success_rate": float(np.mean(recent_succ[-100:])) if recent_succ else float("nan"),
                                    **timer.stats(), **getattr(run_upper, "_last", {})})

        last_value = algo.value(to_t(exo))
        buffer.compute_returns(last_value, gamma, lam)
        run_upper._last = algo.update(buffer)

    ckpt = Path("dhrl/results") / f"{exp_name}_seed_{cfg.get('seed', 0)}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(), "cfg": cfg}, ckpt)
    for env in envs:
        env.close()
    return str(ckpt)


def train_upper(cfg: dict):
    """Build the Habitat-backed task envs from the config, then run the IPPO loop."""
    from dhrl.envs.corridor_crossing import CorridorCrossing
    from dhrl.envs.habitat_backend import HabitatBackend

    task = {"corridor_crossing": CorridorCrossing}[cfg["task"]]
    n_agents = cfg.get("n_agents", 2)
    config_name = cfg.get("config_name", "benchmark/multi_agent/replica_cad_spot_spot.yaml")

    def make(i):
        backend = HabitatBackend(config_name=config_name, seed=cfg["seed"] + i, n_agents=n_agents)
        return task(backend, proprio_dim=backend.proprio_dim, command_dim=cfg.get("command_dim", 3),
                    action_dim=backend.action_dim, rng=np.random.default_rng(cfg["seed"] + i))

    n_envs = cfg.get("num_envs", 1)
    return run_upper([(lambda i=i: make(i)) for i in range(n_envs)], cfg,
                     exp_name=f"{cfg.get('algorithm', 'ippo')}_{cfg['task']}")
