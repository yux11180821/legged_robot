# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""STAGE 2 training loop (paper §5.2.2): IPPO over UL+ML with the LL frozen.

One ``train(args, envs, agent, buffer)`` function, read top-to-bottom (MARL_for_MeltingPot
style): collect a rollout from E parallel cooperation envs with ``agent.take_action`` ->
``env.step`` -> ``buffer.insert``, then ``agent.update`` (truncation-aware GAE + clipped
PPO).  Envs are built in ``run.py`` and passed in.
"""
import os

import numpy as np
import torch

from utils.logging import InferenceTimer, StepLogger


def train(args, envs, agent, buffer):
    """Run stage-2 IPPO and save the actor+critic.

    Args:
        args: flat config (rollout_steps, total_steps, run_dir, seed, log_every, ...).
        envs: list of E cooperation envs (each exposes reset/step/get_env_info/n_agents).
        agent: an ``IPPO`` learner.
        buffer: a ``RolloutBuffer`` sized [rollout_steps, E, N].
    Returns:
        path to the saved checkpoint.
    """
    device = torch.device(getattr(args, "device", "cpu"))
    E, N = len(envs), envs[0].n_agents
    run_dir = getattr(args, "run_dir", "results")
    os.makedirs(run_dir, exist_ok=True)
    tag = f"{args.algo.split('.')[-1]}_{getattr(args, 'task', 'task')}_seed{args.seed}"
    logger = StepLogger(os.path.join(run_dir, f"{tag}_metrics.csv"),
                        log_every=getattr(args, "log_every", 2000),
                        fields=["env_steps", "reward", "success_rate", "infer_ms_median",
                                "policy_loss", "value_loss", "entropy"])
    timer = InferenceTimer(device=str(device))

    def to_t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    # --- reset all envs into an [E, N, *] batch ---
    resets = [env.reset() for env in envs]
    exo = np.stack([r[0] for r in resets])
    proprio = np.stack([r[1] for r in resets])
    hidden = agent.init_hidden(E)
    total_steps, recent_succ, last_losses = 0, [], {}

    while total_steps < args.total_steps:
        buffer.reset()
        # --- collect one rollout ---
        for _ in range(args.rollout_steps):
            with timer.measure():
                out = agent.take_action(to_t(exo), to_t(proprio), hidden)
            actions = out["env_action"].cpu().numpy()

            next_exo, next_pro = np.zeros_like(exo), np.zeros_like(proprio)
            reward = np.zeros((E, N), np.float32)
            done = np.zeros(E, np.float32)
            boot = np.zeros((E, N), np.float32)
            for e, env in enumerate(envs):
                ne, npro, r, d, info = env.step(actions[e])
                reward[e] = r
                if d:
                    done[e] = 1.0
                    if info.get("truncated"):                       # time-limit -> bootstrap V(term)
                        boot[e] = agent.get_value(to_t(info["term_exo"])[None])[0].cpu().numpy()
                    else:                                           # true terminal -> record success
                        recent_succ.append(1.0 if info.get("success") else 0.0)
                    ne, npro = env.reset()
                next_exo[e], next_pro[e] = ne, npro

            buffer.insert(exo=to_t(exo), hidden=hidden if agent.use_memory else None,
                          raw_command=out["raw_command"], log_prob=out["log_prob"],
                          value=out["value"], reward=to_t(reward), done=to_t(done),
                          bootstrap=to_t(boot))
            exo, proprio = next_exo, next_pro
            hidden = out["next_hidden"]
            if hidden is not None:                                  # fresh memory for reset envs
                for e in range(E):
                    if done[e] > 0.5:
                        hidden[e].zero_()
            total_steps += E
            if logger.should_log(total_steps):
                logger.log(total_steps, {
                    "reward": float(reward.mean()),
                    "success_rate": float(np.mean(recent_succ[-100:])) if recent_succ else float("nan"),
                    **timer.stats(), **last_losses})

        # --- learn from the rollout ---
        last_value = agent.get_value(to_t(exo))
        last_losses = agent.update(buffer, last_value)

    ckpt = os.path.join(run_dir, f"{tag}.pt")
    agent.save(ckpt)
    for env in envs:
        env.close()
    return ckpt
