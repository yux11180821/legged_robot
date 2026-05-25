import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from habitat.vector_env import VectorEnv

# 初始化分布式进程
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # 双卡 A800：单卡可以轻松支撑 64+ 个并行环境
    num_envs_per_card = 64
    envs = VectorEnv(
        make_env_fn=make_unitree_env,
        num_environments=num_envs_per_card
    )

    # 初始化你的 ActorCriticPPO 模型 (复用之前的网络结构)
    model = ActorCriticPPO(obs_shape=(4, 84, 84), action_dim=12).to(device)

    # 包装为 DDP 模型，跨卡同步梯度
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # 超长 Rollout 以克服局部最优
    rollout_steps = 256

    print(f"Rank {local_rank} 准备就绪，开始收集数据...")

    for update in range(total_updates):
        # 1. 经验收集 (Rollout)
        # 这里的张量分配会直接利用到 80GB 的大显存
        states, actions, log_probs, rewards, dones = collect_rollouts(envs, model, rollout_steps)

        # 2. 计算 GAE (Generalized Advantage Estimation)
        advantages, returns = compute_gae(rewards, values, dones)

        # 3. PPO 更新
        for epoch in range(ppo_epochs):
            # 数据切分与小批量训练
            loss = ppo_update(model, optimizer, states, actions, log_probs, advantages, returns)

        # 只有主进程负责打印日志和保存模型
        if local_rank == 0 and update % log_interval == 0:
            print(f"Update: {update}, Loss: {loss.item()}")
            # 记录到 TensorBoard 或 WandB

if __name__ == "__main__":
    main()