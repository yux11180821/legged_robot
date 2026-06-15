# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-logic self-tests for the D-HRL reproduction (no Habitat / no GPU).

Run:  python dhrl_habitat/selftest.py
Covers: ego-frame geometry (hand-checked against habitat conventions), the
three-layer policies' log-prob consistency, manual commands, the CTDE MARL
update (shapes + the policy actually improves on a synthetic task), and the
team-advantage broadcast.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhrl_habitat.geometry import (  # noqa: E402
    heading_delta_to,
    manual_velocity_command,
    rel_target_ego,
    world_to_ego,
)
from dhrl_habitat.layers import (  # noqa: E402
    ACTION_DIM,
    COMMAND_DIM,
    EXO_OBS_DIM,
    LOWER_OBS_DIM,
    CentralCritic,
    DHRLActor,
    FlatActor,
    LowerOperator,
    build_actor,
    build_lower_obs,
)
from dhrl_habitat.marl_ppo import (  # noqa: E402
    MARLBatch,
    broadcast_team_advantage,
    marl_ppo_update,
)


def test_geometry() -> None:
    # Convention recap: heading h => forward axis (cos h, -sin h) in world (x, z);
    # ego_lateral axis = local -Z = (-sin h, -cos h); wz>0 increases h.
    # Case 1: h=0, displacement along +x (straight ahead).
    f, l = world_to_ego(1.0, 0.0, 0.0)
    assert abs(f - 1.0) < 1e-6 and abs(l) < 1e-6
    # Case 2: h=0, displacement along +z. Forward=(1,0): +z is to the agent's
    # NEGATIVE lateral side (lateral axis is -z), and facing it needs wz<0.
    f, l = world_to_ego(0.0, 1.0, 0.0)
    assert abs(f) < 1e-6 and abs(l + 1.0) < 1e-6
    assert heading_delta_to(f, l) < 0
    # Case 3: h=pi/2 => forward = (0,-1) (the -z direction).
    f, l = world_to_ego(0.0, -2.0, np.pi / 2)
    assert abs(f - 2.0) < 1e-6 and abs(l) < 1e-6
    # Case 4: h=pi/2, target at world -x => ego lateral +1 (turn left/wz>0).
    f, l = world_to_ego(-1.0, 0.0, np.pi / 2)
    assert abs(f) < 1e-6 and abs(l - 1.0) < 1e-6
    assert heading_delta_to(f, l) > 0
    # rel_target_ego end-to-end with a LocalizationSensor-style vector.
    loc = np.array([2.0, 0.1, 3.0, 0.0])  # at (x=2, z=3), facing +x
    ef, el, d = rel_target_ego(loc, np.array([4.0, 3.0]))
    assert abs(ef - 2.0) < 1e-5 and abs(el) < 1e-5 and abs(d - 2.0) < 1e-5
    print("[ok] geometry: ego transforms + turn signs match habitat conventions")


def test_manual_command() -> None:
    cmd = manual_velocity_command(2.0, 0.0, 2.0)
    assert cmd[0] > 0.4 and abs(cmd[1]) < 1e-6 and abs(cmd[2]) < 1e-6  # straight ahead
    cmd = manual_velocity_command(0.0, 1.5, 1.5)
    assert cmd[2] > 0.9  # target on +lateral side -> strong positive turn
    cmd = manual_velocity_command(1.0, 0.0, 0.1)
    assert np.allclose(cmd, 0.0)  # inside stop_dist -> stop
    assert np.all(np.abs(manual_velocity_command(5.0, -5.0, 7.0)) <= 1.0)
    print("[ok] manual command: direction, turn sign, stop, bounds")


def test_layers() -> None:
    torch.manual_seed(0)
    B = 6
    # Lower operator: 3-arg evaluate_actions matches ppo_update's contract.
    lower = LowerOperator()
    obs = torch.randn(B, LOWER_OBS_DIM)
    raw, env_a, lp, val = lower.act(obs)
    assert env_a.shape == (B, ACTION_DIM) and torch.all(env_a.abs() < 1.0)
    lp2, ent, val2 = lower.evaluate_actions(obs, None, raw)
    assert torch.allclose(lp, lp2, atol=1e-5)

    # DHRLActor (UL+ML GRU) and FlatActor share the act/evaluate API.
    for actor, has_mem in ((DHRLActor(), True), (DHRLActor(use_memory=False), False), (FlatActor(), False)):
        exo = torch.randn(B, EXO_OBS_DIM)
        hid = actor.initial_hidden(B, torch.device("cpu"))
        assert (hid is not None) == has_mem
        raw, cmd, lp, nxt = actor.act(exo, hid)
        assert cmd.shape == (B, COMMAND_DIM) and torch.all(cmd.abs() < 1.0)
        lp2, ent = actor.evaluate_actions(exo, hid, raw)
        assert torch.allclose(lp, lp2, atol=1e-5), type(actor).__name__
        if has_mem:
            assert nxt is not None and nxt.shape == (B, actor.hidden_size)

    # build_actor registry + lower-obs builder.
    for name, hierarchical in (("dhrl", True), ("no_memory", True), ("no_hierarchy", False)):
        actor, spec = build_actor(name)
        assert spec.hierarchical == hierarchical
    lo = build_lower_obs(torch.zeros(B, COMMAND_DIM), torch.zeros(B, ACTION_DIM))
    assert lo.shape == (B, LOWER_OBS_DIM)
    print("[ok] layers: shapes, (-1,1) outputs, log-prob consistency, registry")


def test_marl_update_improves() -> None:
    """The CTDE update must increase the probability of positive-advantage actions."""
    torch.manual_seed(0)
    N, T, E = 2, 16, 4
    S, EE = T * E * N, T * E
    actor, _ = build_actor("dhrl", hidden_size=32)
    critic = CentralCritic(N * EXO_OBS_DIM, hidden_size=32)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-3)

    exo = torch.randn(S, EXO_OBS_DIM)
    hid = torch.zeros(S, 32)
    with torch.no_grad():
        raw, cmd, lp, _ = actor.act(exo, hid)
    # Synthetic objective: reward pushing command[0] positive -> advantage = sign(cmd_x)
    adv = torch.sign(cmd[:, 0]).float()
    batch = MARLBatch(
        actor_obs=exo, actor_hidden=hid, raw_actions=raw, old_log_probs=lp,
        advantages=adv,
        critic_obs=torch.randn(EE, N * EXO_OBS_DIM),
        returns=torch.randn(EE),
    )
    losses = marl_ppo_update(actor, critic, opt, batch, ppo_epochs=8, num_minibatches=2)
    with torch.no_grad():
        new_cmd, _ = actor.deterministic_command(exo, hid)
    before = float(torch.sign(cmd[:, 0]).float().mean())
    after = float(torch.sign(new_cmd[:, 0]).float().mean())
    assert after > before + 0.2, (before, after)
    assert all(np.isfinite(v) for v in losses.values())
    print(f"[ok] marl update: P(cmd_x>0) sign-mean {before:+.2f} -> {after:+.2f}; "
          f"losses finite")


def test_broadcast() -> None:
    team = np.arange(6, dtype=np.float32).reshape(3, 2)  # [T=3, E=2]
    per_agent = broadcast_team_advantage(team, 4)
    assert per_agent.shape == (3, 2, 4)
    assert np.all(per_agent[1, 0] == team[1, 0])
    print("[ok] team-advantage broadcast")


def test_exo_dim_consistency() -> None:
    # multi_env builds: 3 goal terms + 3 neighbor terms + COMMAND_DIM last-cmd.
    assert EXO_OBS_DIM == 3 + 3 + COMMAND_DIM
    print("[ok] EXO_OBS_DIM consistent with the adapter layout")


if __name__ == "__main__":
    test_geometry()
    test_manual_command()
    test_layers()
    test_marl_update_improves()
    test_broadcast()
    test_exo_dim_consistency()
    print("\nALL D-HRL SELF-TESTS PASSED")
