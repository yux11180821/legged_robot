# 面向具身协作的分布式层级强化学习复现

本仓库用于整理和复现论文：

**Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation**
（`2407.06499v1.pdf`）。

核心实现位于 `dhrl/`。这里实现的不是 single-RL baseline，而是论文中的分布式层级控制器：

- **上层模块（Upper Layer, UL）**：编码每个智能体的外感知观测，包括任务相关特征和最近邻智能体信息。
- **中层模块（Middle Layer, ML）**：带时空记忆的循环指令策略，用于建模协作任务中的顺序逻辑。
- **下层模块（Lower Layer, LL）**：运动指令接口。在当前 proxy 环境中，它是速度/位置指令适配器；在 IsaacSim/Gym 中，应接入预训练的 Ant 或腿式机器人低层运动控制策略。

代码同时包含论文中的消融设置：

- `dhrl`：带 recurrent memory 的分布式层级策略。
- `no_memory`：去掉中层 RNN 记忆的层级策略。
- `no_hierarchy`：单层 flat MLP 策略，对应无层级结构的消融。

## 快速开始

在带 PyTorch 的环境中安装依赖：

```bash
pip install -r requirements.txt
```

运行一个快速 proxy 复现实验：

```bash
python -m dhrl.train --config configs/dhrl/cooperative_transport.json --variant dhrl
python -m dhrl.train --config configs/dhrl/cooperative_transport.json --variant no_memory
python -m dhrl.train --config configs/dhrl/cooperative_transport.json --variant no_hierarchy
```

运行论文中的三个协作场景：

```bash
bash scripts/run_dhrl_proxy_4seeds.sh
```

绘制论文风格的均值/阴影曲线：

```bash
python scripts/plot_dhrl_shadow.py \
  --log-dir results/dhrl_proxy/cooperative_transport \
  --metric success_rate \
  --out results/dhrl_proxy/cooperative_transport_shadow.png
```

## IsaacSim/Gym 说明

论文中的最终数值需要 IsaacSim/Gym 环境和预训练的低层 locomotion controller。本地项目快照中没有包含这些仿真资产和低层控制器权重。当前包含的 proxy 环境用于复现**算法结构**和消融流程；`dhrl/isaac_adapter.py` 记录了把同一套策略接入 IsaacSim/Gym 任务时需要满足的观测/动作接口。

## 仓库内容

- `dhrl/models.py`：UL/ML/LL actor-critic 策略，以及对应消融模型。
- `dhrl/ippo.py`：用于分布式多智能体控制的 Independent PPO 训练流程。
- `dhrl/proxy_envs.py`：Cooperative Transport、Corridor Crossing、Ravine Bridging 三个 proxy 任务。
- `configs/dhrl/*.json`：按照论文场景整理的配置文件。
- `scripts/run_dhrl_proxy_4seeds.sh`：多 seed 运行脚本，用于生成阴影图数据。
- `scripts/plot_dhrl_shadow.py`：均值/标准差阴影曲线绘图脚本。
