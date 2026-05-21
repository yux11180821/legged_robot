# D-HRL 复现说明

本项目现在包含 `2407.06499v1` 论文对应的 D-HRL 复现代码，而不只是之前用于 PointNav 曲线的 single-RL PPO baseline。

## 论文模块与代码对应关系

| 论文模块 | 代码位置 |
| --- | --- |
| 上层模块：外感知特征提取 | `dhrl.models.DistributedHierarchicalActorCritic.upper` |
| 中层模块：RNN 时空记忆 | `dhrl.models.DistributedHierarchicalActorCritic.middle` |
| 下层模块：预训练 locomotion controller 接口 | `dhrl.models.LowerLayerCommandAdapter`, `dhrl.isaac_adapter` |
| 分布式可扩展的局部观测 | `dhrl.proxy_envs.BaseProxyEnv._nearest_rel` |
| IPPO 多智能体优化 | `dhrl.ippo.train` |
| 无层级结构消融 | `dhrl.models.NoHierarchyActorCritic` |
| 无时空记忆消融 | `dhrl.models.NoMemoryActorCritic` |
| Cooperative Transport 场景 | `dhrl.proxy_envs.CooperativeTransportEnv` |
| Corridor Crossing 场景 | `dhrl.proxy_envs.CorridorCrossingEnv` |
| Ravine Bridging 场景 | `dhrl.proxy_envs.RavineBridgingEnv` |

## 距离论文精确数值还需要什么

论文报告的 success rate 需要以下内容：

1. IsaacSim/Gym 中三个具身协作环境的完整资产。
2. 预训练的 Ant 低层 locomotion controller。
3. 与论文一致的物理 reset 分布和 reward 系数。

当前 proxy 任务刻意保持轻量，目的是在没有 Isaac 的情况下跑通算法结构和消融流程。因此它们应被视为可执行的复现脚手架，而不是最终的 Isaac benchmark 数值。
