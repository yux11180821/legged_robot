# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-logic self-tests for the Commander core (no Habitat / no GPU needed).

Run:  python habitat_commander/selftest.py
Validates the two bug fixes:
  (1) truncation-aware GAE bootstraps time-limit truncations but not true terminals;
  (2) Tanh-squashed policy emits actions in (-1, 1) with a consistent log-prob.
"""

from __future__ import annotations

import numpy as np

from ppo import compute_gae_truncated


def test_gae_truncation_vs_terminal() -> None:
    # Single env, 1 step, reward 0, V(s_t)=0 so the advantage == the bootstrap term.
    gamma, lam = 0.99, 0.95
    rewards = np.array([[0.0]], dtype=np.float32)
    values = np.array([[0.0]], dtype=np.float32)
    last_value = np.array([0.0], dtype=np.float32)
    dones = np.array([[1.0]], dtype=np.float32)  # episode ends at t=0

    # True terminal: bootstrap value 0 -> advantage 0.
    adv_term, ret_term = compute_gae_truncated(
        rewards, values, dones, np.array([[0.0]], np.float32), last_value, gamma, lam
    )
    # Time-limit truncation: bootstrap value V(terminal)=5 -> advantage = gamma*5.
    adv_trunc, ret_trunc = compute_gae_truncated(
        rewards, values, dones, np.array([[5.0]], np.float32), last_value, gamma, lam
    )

    assert abs(adv_term[0, 0] - 0.0) < 1e-6, adv_term
    assert abs(adv_trunc[0, 0] - gamma * 5.0) < 1e-6, adv_trunc
    # The whole point: truncation must NOT equal terminal.
    assert not np.allclose(adv_term, adv_trunc)
    print(f"[ok] GAE truncation: terminal adv={adv_term[0,0]:.4f}  "
          f"truncated adv={adv_trunc[0,0]:.4f} (= gamma*5 = {gamma*5:.4f})")


def test_gae_no_cross_episode_leak() -> None:
    # 3 steps, episode boundary at t=1 (done). Advantage at t=0 must not see t=2.
    gamma, lam = 0.99, 0.95
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    dones = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)  # terminal at t=1
    boots = np.zeros((3, 1), dtype=np.float32)  # true terminal -> 0 bootstrap
    last_value = np.array([0.0], dtype=np.float32)
    adv, ret = compute_gae_truncated(rewards, values, dones, boots, last_value, gamma, lam)
    # t=1 is terminal: adv[1] = r1 = 1.0 (no bootstrap, no future).
    assert abs(adv[1, 0] - 1.0) < 1e-6, adv
    # t=0: delta0 + gamma*lam*(1-done0)*last_gae(=adv at t1) = 1 + gamma*lam*1.0
    expected0 = 1.0 + gamma * lam * 1.0
    assert abs(adv[0, 0] - expected0) < 1e-6, (adv[0, 0], expected0)
    print(f"[ok] GAE no cross-episode leak: adv[t0]={adv[0,0]:.4f} (= 1+gamma*lam), adv[t1]={adv[1,0]:.4f}")


def test_tanh_policy() -> None:
    try:
        import torch
        from commander_policy import CommanderPolicy, TanhNormal
    except Exception as e:  # torch not installed on this machine
        print(f"[skip] torch unavailable ({type(e).__name__}); policy test skipped")
        return
    torch.manual_seed(0)
    policy = CommanderPolicy(obs_dim=16, action_dim=3, hidden_size=32, use_memory=True)
    obs = torch.randn(8, 16)
    hidden = policy.initial_hidden(8, torch.device("cpu"))
    raw, env_action, log_prob, value, next_hidden = policy.act(obs, hidden)

    assert env_action.shape == (8, 3)
    assert torch.all(env_action > -1.0) and torch.all(env_action < 1.0), "action must be in (-1,1)"
    assert torch.isfinite(log_prob).all() and log_prob.shape == (8,)
    assert next_hidden is not None and next_hidden.shape == (8, 32)

    # Ratio self-consistency: re-evaluating the stored raw action reproduces log_prob.
    lp2, ent, val2 = policy.evaluate_actions(obs, hidden, raw)
    max_delta = float((log_prob - lp2).abs().max())
    assert torch.allclose(log_prob, lp2, atol=1e-5), max_delta
    print(f"[ok] tanh policy: env_action in ({float(env_action.min()):.3f},{float(env_action.max()):.3f}); "
          f"log_prob consistent (max|delta|={max_delta:.2e})")


def test_metrics_harness() -> None:
    import os
    import tempfile

    from metrics import InferenceTimer, StepLogger

    # Inference timer records a positive latency.
    timer = InferenceTimer(device="cpu")
    with timer.measure():
        _ = sum(i * i for i in range(10000))
    st = timer.stats()
    assert st["infer_ms_last"] >= 0.0 and st["infer_ms_mean"] >= 0.0

    # StepLogger logs exactly once per --log-every bucket, even with vectorized jumps.
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "m.csv")
        logged = []
        logger = StepLogger(csv_path, log_every=20, print_fn=lambda *_: None)
        # num_envs=8 -> frames jump 8,16,24,32,40,...; buckets of 20 crossed at 24, 40, ...
        for frames in range(8, 81, 8):
            if logger.maybe_log(frames, {"reward": 1.0, **timer.stats()}):
                logged.append(frames)
        # frames=8,16,24,...,80 with buckets of 20 -> first row of each new bucket:
        # bucket0@8, bucket1@24, bucket2@40, bucket3@64, bucket4@80.
        assert logged == [8, 24, 40, 64, 80], logged
        # No double logging within a bucket, no skipped bucket.
        rows = open(csv_path).read().strip().splitlines()
        assert len(rows) == 1 + len(logged), rows  # header + one row each
    print(f"[ok] metrics: inference timer ok; StepLogger buckets={logged} (one row per 20-step bucket)")


if __name__ == "__main__":
    test_gae_truncation_vs_terminal()
    test_gae_no_cross_episode_leak()
    test_tanh_policy()
    test_metrics_harness()
    print("\nALL SELF-TESTS PASSED")
