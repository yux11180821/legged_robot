from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .config import DHRLConfig


@dataclass
class ProxyObs:
    proprio: np.ndarray
    extero: np.ndarray


class BaseProxyEnv:
    """Vectorized lightweight proxy for the paper's embodied cooperation tasks."""

    def __init__(self, config: DHRLConfig) -> None:
        self.config = config
        self.num_envs = config.num_envs
        self.num_agents = config.num_agents
        self.max_episode_steps = config.max_episode_steps
        self.rng = np.random.default_rng(config.seed)
        self.t = np.zeros(self.num_envs, dtype=np.int32)
        self.agent_pos = np.zeros((self.num_envs, self.num_agents, 2), dtype=np.float32)
        self.agent_vel = np.zeros_like(self.agent_pos)
        self.object_pos = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.target_pos = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.bridge_pos = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.last_success = np.zeros(self.num_envs, dtype=bool)
        self.reset()

    def reset(self) -> ProxyObs:
        self.t.fill(0)
        self._reset_indices(np.arange(self.num_envs))
        return self._obs()

    def _reset_indices(self, ids: np.ndarray) -> None:
        raise NotImplementedError

    def step(self, actions: np.ndarray) -> tuple[ProxyObs, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        reward, success = self._dynamics(actions)
        self.t += 1
        done_env = np.logical_or(success, self.t >= self.max_episode_steps)
        done = np.repeat(done_env[:, None], self.num_agents, axis=1)
        self.last_success = success
        if done_env.any():
            self._reset_indices(np.where(done_env)[0])
        info = {
            "success": success.astype(np.float32),
            "success_rate": float(success.mean()),
            "episode_done": done_env.astype(np.float32),
        }
        return self._obs(), reward.astype(np.float32), done.astype(np.float32), info

    def _dynamics(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _nearest_rel(self) -> np.ndarray:
        rel = self.agent_pos[:, :, None, :] - self.agent_pos[:, None, :, :]
        dist = np.linalg.norm(rel, axis=-1)
        dist += np.eye(self.num_agents, dtype=np.float32)[None] * 1e6
        idx = dist.argmin(axis=-1)
        nearest = self.agent_pos[np.arange(self.num_envs)[:, None], idx]
        return nearest - self.agent_pos

    def _obs(self) -> ProxyObs:
        nearest_rel = self._nearest_rel()
        to_object = self.object_pos[:, None, :] - self.agent_pos
        to_target = self.target_pos[:, None, :] - self.agent_pos
        bridge_rel = self.bridge_pos[:, None, :] - self.agent_pos
        proprio = np.concatenate(
            [
                self.agent_pos / 10.0,
                self.agent_vel,
                to_target / 10.0,
                np.linalg.norm(to_target, axis=-1, keepdims=True) / 10.0,
                np.ones((self.num_envs, self.num_agents, 1), dtype=np.float32),
            ],
            axis=-1,
        )
        extero = np.concatenate(
            [
                nearest_rel / 10.0,
                to_object / 10.0,
                to_target / 10.0,
                bridge_rel / 10.0,
                self._task_features(),
            ],
            axis=-1,
        )
        return ProxyObs(proprio.astype(np.float32), extero.astype(np.float32))

    def _task_features(self) -> np.ndarray:
        return np.zeros((self.num_envs, self.num_agents, 4), dtype=np.float32)


class CooperativeTransportEnv(BaseProxyEnv):
    def _reset_indices(self, ids: np.ndarray) -> None:
        n = len(ids)
        angles = np.linspace(0, 2 * math.pi, self.num_agents, endpoint=False)
        radii = self.rng.uniform(3.0, 7.0, size=(n, self.num_agents, 1))
        jitter = self.rng.normal(0.0, 0.4, size=(n, self.num_agents, 2))
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=-1)[None]
        self.object_pos[ids] = self.rng.normal(0.0, 0.4, size=(n, 2))
        self.target_pos[ids] = self.rng.uniform(-6.0, 6.0, size=(n, 2))
        self.agent_pos[ids] = self.object_pos[ids, None, :] + radii * circle + jitter
        self.agent_vel[ids] = 0.0
        self.bridge_pos[ids] = 0.0
        self.t[ids] = 0

    def _dynamics(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.agent_vel = 0.9 * self.agent_vel + 0.25 * actions
        self.agent_pos += self.agent_vel
        rel_obj = self.object_pos[:, None, :] - self.agent_pos
        dist_obj = np.linalg.norm(rel_obj, axis=-1)
        contact = dist_obj < 1.8
        push = (actions * contact[..., None]).sum(axis=1)
        self.object_pos += 0.18 * np.tanh(push)
        object_to_target = np.linalg.norm(self.object_pos - self.target_pos, axis=-1)
        agent_to_object = dist_obj.mean(axis=1)
        agent_to_target = np.linalg.norm(self.agent_pos - self.target_pos[:, None, :], axis=-1).mean(axis=1)
        env_reward = 2.0 / (1.0 + object_to_target) + 0.25 / (1.0 + agent_to_object) + 0.05 / (1.0 + agent_to_target)
        success = object_to_target < 0.6
        reward = np.repeat(env_reward[:, None], self.num_agents, axis=1)
        reward += success[:, None] * 5.0
        return reward, success


class CorridorCrossingEnv(BaseProxyEnv):
    def _reset_indices(self, ids: np.ndarray) -> None:
        n = len(ids)
        xs = np.linspace(-self.num_agents + 1, self.num_agents - 1, self.num_agents)
        self.agent_pos[ids, :, 0] = xs[None] + self.rng.normal(0.0, 0.15, size=(n, self.num_agents))
        self.agent_pos[ids, :, 1] = self.rng.uniform(-7.0, -5.5, size=(n, self.num_agents))
        self.agent_vel[ids] = 0.0
        self.object_pos[ids] = np.array([0.0, 0.0], dtype=np.float32)
        self.target_pos[ids] = np.array([0.0, 7.5], dtype=np.float32)
        self.bridge_pos[ids] = np.array([0.0, 0.0], dtype=np.float32)
        self.t[ids] = 0

    def _dynamics(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        next_vel = 0.85 * self.agent_vel + 0.22 * actions
        next_pos = self.agent_pos + next_vel
        near_wall = np.abs(next_pos[:, :, 1]) < 0.7
        outside_slit = np.abs(next_pos[:, :, 0]) > 0.75
        blocked = near_wall & outside_slit
        next_pos[blocked] = self.agent_pos[blocked]
        next_vel[blocked] *= -0.2
        pair = self.agent_pos[:, :, None, :] - self.agent_pos[:, None, :, :]
        collision = (np.linalg.norm(pair, axis=-1) < 0.7).sum(axis=-1) > 1
        self.agent_pos = next_pos
        self.agent_vel = next_vel
        passed = self.agent_pos[:, :, 1] > 7.0
        enter_dist = np.linalg.norm(self.agent_pos - np.array([0.0, -0.2], dtype=np.float32), axis=-1)
        reward = 0.4 * passed.astype(np.float32) + 0.05 / (1.0 + enter_dist) - 0.04 * collision.astype(np.float32)
        success = passed.all(axis=1)
        reward += success[:, None] * 4.0
        return reward, success

    def _task_features(self) -> np.ndarray:
        wall = np.array([0.75, 0.0, 7.0, 1.0], dtype=np.float32)
        return np.repeat(wall[None, None, :], self.num_envs, axis=0).repeat(self.num_agents, axis=1)


class RavineBridgingEnv(BaseProxyEnv):
    def _reset_indices(self, ids: np.ndarray) -> None:
        n = len(ids)
        xs = np.linspace(-self.num_agents + 1, self.num_agents - 1, self.num_agents)
        self.agent_pos[ids, :, 0] = xs[None] + self.rng.normal(0.0, 0.2, size=(n, self.num_agents))
        self.agent_pos[ids, :, 1] = self.rng.uniform(-6.0, -4.0, size=(n, self.num_agents))
        self.agent_vel[ids] = 0.0
        self.object_pos[ids] = np.array([0.0, 0.0], dtype=np.float32)
        self.bridge_pos[ids] = np.array([0.0, -1.5], dtype=np.float32)
        self.target_pos[ids] = np.array([0.0, 7.0], dtype=np.float32)
        self.t[ids] = 0

    def _dynamics(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bridge_agents = min(2, self.num_agents)
        self.agent_vel = 0.86 * self.agent_vel + 0.18 * actions
        self.agent_pos += self.agent_vel
        pushers = self.agent_pos[:, :bridge_agents]
        pusher_dist = np.linalg.norm(pushers - self.bridge_pos[:, None, :], axis=-1)
        contact = pusher_dist < 1.2
        bridge_push = (actions[:, :bridge_agents] * contact[..., None]).sum(axis=1)
        self.bridge_pos += 0.16 * np.tanh(bridge_push)
        ravine_y = 0.0
        bridge_ready = np.linalg.norm(self.bridge_pos - np.array([0.0, ravine_y], dtype=np.float32), axis=-1) < 0.8
        crossing = (np.abs(self.agent_pos[:, :, 1] - ravine_y) < 0.6) & (np.abs(self.agent_pos[:, :, 0]) < 2.0)
        falling = crossing & (~bridge_ready[:, None])
        self.agent_pos[falling, 1] -= 0.8
        passed = self.agent_pos[:, bridge_agents:, 1] > 6.5 if self.num_agents > bridge_agents else np.ones((self.num_envs, 1), dtype=bool)
        bridge_reward = 0.8 / (1.0 + np.linalg.norm(self.bridge_pos - np.array([0.0, ravine_y], dtype=np.float32), axis=-1))
        dest_reward = passed.mean(axis=1)
        reward = np.repeat((bridge_reward + dest_reward)[:, None], self.num_agents, axis=1)
        reward -= 0.2 * falling.astype(np.float32)
        success = passed.all(axis=1) & bridge_ready
        reward += success[:, None] * 5.0
        return reward, success

    def _task_features(self) -> np.ndarray:
        ravine = np.array([0.0, 0.0, 2.0, 1.0], dtype=np.float32)
        return np.repeat(ravine[None, None, :], self.num_envs, axis=0).repeat(self.num_agents, axis=1)


def make_env(config: DHRLConfig) -> BaseProxyEnv:
    if config.scenario == "cooperative_transport":
        return CooperativeTransportEnv(config)
    if config.scenario == "corridor_crossing":
        return CorridorCrossingEnv(config)
    if config.scenario == "ravine_bridging":
        return RavineBridgingEnv(config)
    raise ValueError(f"unknown scenario: {config.scenario}")
