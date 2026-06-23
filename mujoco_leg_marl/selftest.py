# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-logic self-tests for the Go2 leg-MARL core (no mujoco needed).

Run:  python mujoco_leg_marl/selftest.py
Covers the per-leg actor + the reused CTDE update on the 4-leg setup.  The
MuJoCo env itself is smoke-tested separately on the training box (see README).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mujoco_leg_marl.policy import CentralCritic, LegActor  # noqa: E402
from dhrl_habitat.marl_ppo import MARLBatch, broadcast_team_advantage, marl_ppo_update  # noqa: E402

PER_LEG_OBS_DIM = 22
STATE_DIM = 52
A = 4  # legs


def test_leg_actor() -> None:
    torch.manual_seed(0)
    B = 8 * A
    actor = LegActor(PER_LEG_OBS_DIM, action_dim=3, hidden_size=32)
    obs = torch.randn(B, PER_LEG_OBS_DIM)
    raw, env_a, lp, nxt = actor.act(obs, None)
    assert env_a.shape == (B, 3) and torch.all(env_a.abs() < 1.0), "leg action must be in (-1,1)"
    assert nxt is None and not actor.use_memory
    lp2, ent = actor.evaluate_actions(obs, None, raw)
    assert torch.allclose(lp, lp2, atol=1e-5), float((lp - lp2).abs().max())
    print(f"[ok] LegActor: action in ({float(env_a.min()):.3f},{float(env_a.max()):.3f}); log_prob consistent")


def test_dims() -> None:
    # per-leg: base_lin(3)+base_ang(3)+grav(3)+cmd(3)+jpos(3)+jvel(3)+foot(1)+lastact(3)
    assert PER_LEG_OBS_DIM == 3 + 3 + 3 + 3 + 3 + 3 + 1 + 3
    # state: ...+ all jpos(12)+ all jvel(12)+ foot(4)+ lastact(12)
    assert STATE_DIM == 3 + 3 + 3 + 3 + 12 + 12 + 4 + 12
    print("[ok] obs/state dims consistent (22 / 52)")


def test_ctde_update_improves() -> None:
    """The reused CTDE MAPPO update must push leg actions the rewarded way."""
    torch.manual_seed(0)
    N, T = 4, 12
    S, E = T * N * A, T * N
    actor = LegActor(PER_LEG_OBS_DIM, hidden_size=32)
    critic = CentralCritic(STATE_DIM, hidden_size=32)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-3)
    obs = torch.randn(S, PER_LEG_OBS_DIM)
    with torch.no_grad():
        raw, env_a, lp, _ = actor.act(obs, None)
    adv = torch.sign(env_a[:, 0])  # reward pushing first joint positive
    batch = MARLBatch(actor_obs=obs, actor_hidden=None, raw_actions=raw, old_log_probs=lp,
                      advantages=adv, critic_obs=torch.randn(E, STATE_DIM), returns=torch.randn(E))
    losses = marl_ppo_update(actor, critic, opt, batch, ppo_epochs=8, num_minibatches=2, target_kl=None)
    with torch.no_grad():
        after = float(torch.sign(actor.deterministic_action(obs)[:, 0]).mean())
    before = float(torch.sign(env_a[:, 0]).float().mean())
    assert after > before + 0.2, (before, after)
    assert all(np.isfinite(v) for v in losses.values())
    print(f"[ok] CTDE update: P(joint0>0) {before:+.2f} -> {after:+.2f}; losses finite")


def test_broadcast() -> None:
    team = np.arange(6, dtype=np.float32).reshape(3, 2)  # [T,N]
    out = broadcast_team_advantage(team, A)
    assert out.shape == (3, 2, A) and np.all(out[1, 1] == team[1, 1])
    print("[ok] team-advantage broadcast to 4 legs")


if __name__ == "__main__":
    test_dims()
    test_leg_actor()
    test_ctde_update_improves()
    test_broadcast()
    print("\nALL GO2-LEG-MARL SELF-TESTS PASSED")
