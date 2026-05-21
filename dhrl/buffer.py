from __future__ import annotations

import torch


class MultiAgentRolloutBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, num_agents: int, proprio_dim: int, extero_dim: int, action_dim: int, hidden_dim: int, device: torch.device) -> None:
        shape = (rollout_steps, num_envs, num_agents)
        self.rollout_steps = rollout_steps
        self.device = device
        self.proprio = torch.zeros(*shape, proprio_dim, device=device)
        self.extero = torch.zeros(*shape, extero_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.log_probs = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, device=device)
        self.values = torch.zeros(*shape, device=device)
        self.hiddens = torch.zeros(*shape, hidden_dim, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.returns = torch.zeros(*shape, device=device)
        self.step = 0

    def add(self, proprio: torch.Tensor, extero: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor, hidden: torch.Tensor) -> None:
        i = self.step
        self.proprio[i].copy_(proprio)
        self.extero[i].copy_(extero)
        self.actions[i].copy_(action)
        self.log_probs[i].copy_(log_prob)
        self.rewards[i].copy_(reward)
        self.dones[i].copy_(done)
        self.values[i].copy_(value)
        self.hiddens[i].copy_(hidden)
        self.step += 1

    def compute_returns(self, next_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        gae = torch.zeros_like(next_value)
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_values = next_value if t == self.rollout_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def minibatches(self, minibatch_size: int):
        total = self.rollout_steps * self.proprio.shape[1] * self.proprio.shape[2]
        indices = torch.randperm(total, device=self.device)
        flat = {
            "proprio": self.proprio.reshape(total, -1),
            "extero": self.extero.reshape(total, -1),
            "actions": self.actions.reshape(total, -1),
            "log_probs": self.log_probs.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
            "values": self.values.reshape(total),
            "hiddens": self.hiddens.reshape(total, -1),
        }
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        for start in range(0, total, minibatch_size):
            mb = indices[start : start + minibatch_size]
            yield {k: v[mb] for k, v in flat.items()}
