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

from dhrl.distributed.ippo import IPPO
from dhrl.distributed.rollout import RolloutBuffer
from dhrl.distributed.critic import DecentralizedCritic
from dhrl.hierarchy.policy import HRLPolicy
from dhrl.hierarchy.lower_layer import LowerLayer
from dhrl.common.logging import InferenceTimer, StepLogger


def run_upper(env_fns, cfg: dict, *, device=None, results_dir="dhrl/results", exp_name="upper"):
    """Run the full stage-2 IPPO loop training the Upper Layer + Middle Layer over the frozen LL.

    Executable form of paper §5.2.2 and the decentralization claim of §4.2/§5.4. Each homogeneous
    agent runs the same 3-layer HRL policy (Fig.2): the UL perceives the partial exteroceptive
    obs e_t (environment + ONLY the nearest-neighbour relative position, the scalability design),
    the ML is a GRU spatiotemporal memory h_t (§4.3), and the FROZEN LL maps proprio + the ML's
    command to the actuator action. The PPO action variable is the ML's command -- only UL+ML
    (and the critic) receive gradients; the LL never does. The learner is IPPO (Independent PPO,
    FULLY decentralized): every agent optimizes a clipped-PPO objective on its OWN observation,
    so the ``DecentralizedCritic`` is per-agent with NO global-state concatenation and NO
    centralized critic (that CTDE/MAPPO variant is only the baseline the paper beats, Fig.5 b/c).
    Returns use truncation-aware GAE so a time-limit cut bootstraps V rather than being treated
    as a true terminal. Taking env *factories* lets the same loop drive the real Habitat task or
    a mock backend in tests.

    Args:
        env_fns (list[callable]): one zero-arg factory per parallel env, each returning a
            cooperative task env exposing ``n_agents`` N, ``exo_dim`` / ``proprio_dim`` /
            ``command_dim`` / ``action_dim``, ``reset`` -> (exo, proprio) and
            ``step(actions[N])`` -> (exo, proprio, reward[N], done, info).
        cfg (dict): run config -- ``ppo`` block (gamma, gae_lambda, lr, clip, entropy_coef, epochs,
            minibatches), ``rollout_steps`` T, ``total_steps``, ``hidden_size`` H, ``use_memory``
            (False -> the 'No Spatiotemporal Memory' ablation), ``lower_ckpt`` (stage-1 operator),
            ``target_kl``, ``seed``, ``log_every``.
        device: torch device; defaults to CUDA when available else CPU.
        results_dir (str): directory for the per-seed metrics CSV.
        exp_name (str): experiment tag used in the metrics-CSV and checkpoint filenames.

    Returns:
        str: path of the saved checkpoint bundling the trained ``policy`` (UL+ML, with the frozen
        LL weights), the ``critic``, and ``cfg``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = cfg.get("ppo", {})
    gamma = ppo.get("gamma", 0.995)                  # reward discount
    lam = ppo.get("gae_lambda", 0.95)                # GAE bias/variance trade-off
    T = cfg.get("rollout_steps", 128)                # steps per env per PPO iteration
    total = cfg.get("total_steps", 1_000_000)        # total env frames budget (counted as E per step)
    H = cfg.get("hidden_size", 256)                  # shared hidden width (UL/ML/critic and GRU state)
    use_memory = cfg.get("use_memory", True)   # False -> 'No Spatiotemporal Memory' ablation (drops the ML GRU)

    envs = [fn() for fn in env_fns]                  # instantiate all parallel envs
    E, N = len(envs), envs[0].n_agents              # E = parallel envs, N = agents per env (homogeneous)
    exo_d, pro_d, cmd_d, act_d = (envs[0].exo_dim, envs[0].proprio_dim,
                                  envs[0].command_dim, envs[0].action_dim)

    # frozen lower operator (stage-1 product); fall back to a fresh frozen LL for smoke
    if cfg.get("lower_ckpt") and Path(cfg["lower_ckpt"]).exists():
        # Reload the stage-1 operator and stop-grad it: the LL is fixed throughout stage 2.
        lower = LowerLayer.load_frozen(cfg["lower_ckpt"], proprio_dim=pro_d, command_dim=cmd_d,
                                       action_dim=act_d, hidden_size=H)
    else:
        # Smoke/test path: a randomly-initialized but still-frozen LL keeps the wiring identical.
        lower = LowerLayer(pro_d, cmd_d, act_d, H).freeze()
    # Homogeneous agents SHARE this one policy's parameters (param sharing, §5.2); the frozen LL is embedded.
    policy = HRLPolicy(exo_d, pro_d, cmd_d, act_d, H, use_memory=use_memory, lower=lower).to(device)
    critic = DecentralizedCritic(exo_d, H).to(device)  # per-agent value on the LOCAL exo obs only (no global state)
    algo = IPPO(policy, critic, lr=ppo.get("lr", 5e-4), clip_param=ppo.get("clip", 0.2),
                entropy_coef=ppo.get("entropy_coef", 1e-3), ppo_epochs=ppo.get("epochs", 4),
                num_minibatches=ppo.get("minibatches", 4), target_kl=cfg.get("target_kl"))
    buffer = RolloutBuffer(T, E, N, exo_d, cmd_d, H, device, use_memory=use_memory)  # stores hidden only if ML is on
    logger = StepLogger(Path(results_dir) / f"{exp_name}_seed_{cfg.get('seed', 0)}_metrics.csv",
                        log_every=cfg.get("log_every", 2000),
                        fields=["env_steps", "reward", "success_rate", "infer_ms_median",
                                "policy_loss", "value_loss", "entropy"])
    timer = InferenceTimer(device=str(device))       # measures per-decision wall-clock (paper's inference-time metric)

    def to_t(x):
        """Convert array-like ``x`` to a float32 tensor on the training ``device``."""
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    resets = [env.reset() for env in envs]
    # Stack per-env resets into batched arrays: exo [E, N, exo_dim], proprio [E, N, proprio_dim].
    exo = np.stack([r[0] for r in resets]); proprio = np.stack([r[1] for r in resets])
    # Initial GRU state for every (env, agent); kept as [E, N, H], or None when the ML is ablated.
    hidden = policy.initial_hidden(E * N, device).view(E, N, H) if use_memory else None
    frames, recent_succ = 0, []                      # global frame counter and recent success-flag window

    while frames < total:
        buffer.reset()                               # clear the on-policy buffer before each rollout
        for _ in range(T):
            with timer.measure():                    # time ONLY the policy forward pass (decentralized inference cost)
                # IPPO action: returns the sampled ML command (raw + env-ready), its log-prob, value, next GRU state.
                out = algo.act(to_t(exo), to_t(proprio), hidden)
            actions = out["env_action"].cpu().numpy()   # [E, N, action_dim] actions the LL produced from the command

            next_exo, next_pro = np.zeros_like(exo), np.zeros_like(proprio)
            reward = np.zeros((E, N), np.float32); done = np.zeros(E, np.float32)
            boot = np.zeros((E, N), np.float32)         # per-agent bootstrap value, filled only on truncation
            for e, env in enumerate(envs):
                ne, npro, r, d, info = env.step(actions[e])  # step env e with its N agents' actions
                reward[e] = r                            # SHARED cooperative reward broadcast to the N agents (Eq.1)
                if d:
                    done[e] = 1.0
                    if info.get("truncated"):
                        # Time-limit truncation: bootstrap V at the true terminal exo (not the reset obs).
                        boot[e] = algo.value(to_t(info["term_exo"])[None])[0].cpu().numpy()
                    else:
                        # Genuine episode end: record whether the cooperative task succeeded.
                        recent_succ.append(1.0 if info.get("success") else 0.0)
                    ne, npro = env.reset()               # auto-reset this env so the rollout stays dense
                next_exo[e], next_pro[e] = ne, npro

            # Store the transition; hidden is passed only when the ML memory is active.
            buffer.insert(exo=to_t(exo), hidden=hidden if use_memory else None,
                          raw_command=out["raw_command"], log_prob=out["log_prob"],
                          value=out["value"], reward=to_t(reward), done=to_t(done),
                          bootstrap=to_t(boot))
            exo, proprio = next_exo, next_pro            # roll observations forward
            hidden = out["next_hidden"]                  # carry the GRU state to the next step
            if hidden is not None:                       # new episode -> fresh memory
                for e in range(E):
                    if done[e] > 0.5:
                        hidden[e].zero_()                # reset memory for envs that just terminated/reset
            frames += E                                  # one step advances E parallel envs
            if logger.should_log(frames):
                logger.log(frames, {"reward": float(reward.mean()),
                                    # success rate over the last 100 completed episodes (NaN until the first finishes)
                                    "success_rate": float(np.mean(recent_succ[-100:])) if recent_succ else float("nan"),
                                    # merge in inference-time stats and the LAST update's PPO losses (see _last below)
                                    **timer.stats(), **getattr(run_upper, "_last", {})})

        last_value = algo.value(to_t(exo))               # bootstrap value for the final state (per env/agent)
        buffer.compute_returns(last_value, gamma, lam)   # truncation-aware GAE over the rollout
        # IPPO update on UL+ML+critic only (LL stays frozen); stash losses on the function for the next log line.
        run_upper._last = algo.update(buffer)

    ckpt = Path("dhrl/results") / f"{exp_name}_seed_{cfg.get('seed', 0)}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Persist trained policy (UL+ML + embedded frozen LL), the per-agent critic, and the cfg for provenance.
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(), "cfg": cfg}, ckpt)
    for env in envs:
        env.close()
    return str(ckpt)


def train_upper(cfg: dict):
    """Build the Habitat-backed task envs from the config, then run the IPPO loop.

    Thin config->runner adapter and the ``stage == 'upper'`` entry point dispatched by
    ``run.main`` (§5.2.2). It selects the cooperative task class, spins up ``num_envs``
    parallel ``HabitatBackend``-backed envs (each with its own seed offset for decorrelated
    parallel rollouts and N homogeneous Spot agents, §5.1), and hands the env factories to
    ``run_upper`` for the decentralized IPPO training loop.

    Args:
        cfg (dict): the loaded config. Reads ``task`` (selects the task class), ``n_agents`` N,
            ``config_name`` (the Habitat multi-agent scene/robot YAML), ``command_dim``,
            ``num_envs`` E, ``seed``, ``algorithm`` (tag), and forwards the rest to ``run_upper``.

    Returns:
        str: path to the saved stage-2 checkpoint (forwarded from ``run_upper``).
    """
    from dhrl.cooperation.corridor_crossing import CorridorCrossing
    from dhrl.cooperation.habitat_world import HabitatBackend

    task = {"corridor_crossing": CorridorCrossing}[cfg["task"]]  # map the config task name to its class
    n_agents = cfg.get("n_agents", 2)
    config_name = cfg.get("config_name", "benchmark/multi_agent/replica_cad_spot_spot.yaml")  # Habitat scene/robot spec

    def make(i):
        """Factory for parallel env ``i``: a Habitat backend + cooperative task, seed-offset by ``i``."""
        # Offset the seed by the env index so parallel rollouts are decorrelated.
        backend = HabitatBackend(config_name=config_name, seed=cfg["seed"] + i, n_agents=n_agents)
        return task(backend, proprio_dim=backend.proprio_dim, command_dim=cfg.get("command_dim", 3),
                    action_dim=backend.action_dim, rng=np.random.default_rng(cfg["seed"] + i))

    n_envs = cfg.get("num_envs", 1)
    # Build E deferred factories (default-arg ``i=i`` binds the index per closure) and run the IPPO loop;
    # ``exp_name`` embeds the algorithm tag + task so different runs write to distinct files.
    return run_upper([(lambda i=i: make(i)) for i in range(n_envs)], cfg,
                     exp_name=f"{cfg.get('algorithm', 'ippo')}_{cfg['task']}")
