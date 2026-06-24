# `dhrl/` — 分布式分层运动控制（arXiv:2407.06499 复现）

复现 **Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation**
（Hong, Huang, Liu — Tsinghua, CoRL 2024）的**策略本身**。代码按标准 on-policy MARL
框架组织（参考 MARL_for_MeltingPot：`algorithms / networks / envs / memories / runners /
utils / configs`）。

> **本目录目前是骨架（每个文件只有 docstring，标注它对应论文哪一节、将放什么）。
> 实现代码后续填充。**

---

## ⚠️ 复现的算法是 IPPO（去中心化），不是 MAPPO/CTDE

这是重读论文后最重要的一点（也是之前代码的核心问题）：

- §5：“**Independent PPO (IPPO)** in a multi-agent setting is employed as the algorithm.”
- §4.2：完全去中心化、部分观测，“**unattainable by CTDE frameworks**”。
- Fig.5(b)(c) “Comparison with **CTDE**”：`Distributed HRL (IPPO)` vs `Centralized
  Training (MAPPO)` → **MAPPO 收敛更慢、agent 多了不收敛**。

**结论：MAPPO / 中心化 critic 是论文要打败的 baseline，不是论文方法。**
本包以 **IPPO 为主方法**（`algorithms/ippo.py`），MAPPO 仅作对照 baseline 保留
（`algorithms/mappo.py`）。

---

## 整体架构：每个 agent 一份三层 HRL（论文 Fig.2）

```
                         ┌──────── 每个 agent 各跑一份（同构共享参数，去中心化）────────┐
  外部观测 e_t ─────────▶│  UL 上层(感知)  ──feature──▶  ML 中层(RNN, 记忆 h_t)        │
  [环境高度图 + 仅最近邻] │                                      │ command(命令)        │
                         │                                      ▼                      │
  本体观测 p_t ──────────────────────────────────────▶  LL 下层(冻结的运动算子) ──▶ a_t │
  [位置/速度/关节力矩]   └──────────────────────────────────────────────────────────────┘
```

| 层 | 文件 | 职责（论文） |
|---|---|---|
| **UL 上层** | `networks/upper_layer.py` | 感知外部观测 e_t → 特征。e_t = 环境感知 + **仅最近邻**相对位置（§4.2, Fig.3，部分观测=可扩展的关键）|
| **ML 中层** | `networks/middle_layer.py` | **RNN**，维护时空记忆 h_t → 输出**命令**给下层（§4.3；无 RNN 消融惨败，Table 1）|
| **LL 下层** | `networks/lower_layer.py` | **预训练后冻结**的运动算子：本体观测 p_t + 命令 → 动作 a_t；位置/速度两模式（§4.1, §5.2.1）|

---

## 两阶段训练流程

```
阶段1 (单智能体)                    阶段2 (多智能体, 去中心化 IPPO)
┌───────────────────────┐         ┌──────────────────────────────────────┐
│ ppo_single 训 LL 下层 │ 冻结    │ IPPO 训 UL + ML(RNN)，LL 保持冻结      │
│ 位置: r=1/L2距离      │ ──────▶ │ 在 协同搬运/走廊穿越/峡谷架桥 上训练   │
│ 速度: r=速度点积      │ (位置)  │ 课程: 早期稀疏+稠密奖励 (附录)         │
└───────────────────────┘         └──────────────────────────────────────┘
  runners/lower_trainer.py          runners/upper_trainer.py
```

- **阶段1**（`runners/lower_trainer.py`，§5.2.1）：单智能体训下层运动算子，**冻结位置模式**。
- **阶段2**（`runners/upper_trainer.py`，§5.2.2）：冻结下层之上，用 **IPPO** 训 UL+ML。
- 同构共享一套策略 + 共享奖励（Eq.1）；异构（峡谷架桥）用按物种的乘积目标（Eq.2）。

---

## 三个任务（论文 Fig.1）

| 任务 | 文件 | 说明 | 论文成功率 |
|---|---|---|---|
| Cooperative Transport | `envs/cooperative_transport.py` | 同构群体协同把物体搬到目标区 | 76.8% |
| Corridor Crossing | `envs/corridor_crossing.py` | 同构，窄走廊轮流通过（让行/牺牲） | 88.4% |
| Ravine Bridging | `envs/ravine_bridging.py` | **异构**（2 套策略），一组推桥让另一组过 | 50.4% |

（消融，Table 1：No Hierarchy 全 0%；No Spatiotemporal Memory ~5–12%。）

---

## 目录结构

```
dhrl/
  run.py                  统一入口：python -m dhrl.run --config configs/xxx.yaml
  configs/                YAML 配置（base + 阶段1 + 三个任务）
  algorithms/
    ippo.py               ★ 论文方法：Independent PPO（去中心化，去中心 critic）
    mappo.py              CTDE baseline（仅对照，论文要打败的对象）
    ppo_single.py         阶段1 单智能体 PPO（训下层算子）
  networks/
    upper_layer.py        UL 感知
    middle_layer.py       ML 时空记忆 RNN
    lower_layer.py        LL 冻结运动算子
    hrl_policy.py         三层拼成一个 agent 的策略
    critic.py             去中心 critic(IPPO) / 中心 critic(MAPPO)
    distributions.py      TanhNormal 动作分布
  envs/
    base.py / obs.py      多智能体接口 + 观测构造（本体 p / 外部 e，仅最近邻）
    habitat_backend.py    仿真后端（论文用 IsaacSim+Ant；我们用 Habitat+Spot）
    cooperative_transport.py / corridor_crossing.py / ravine_bridging.py
  memories/rollout_buffer.py   带 RNN 隐状态 + 截断 GAE 的 on-policy buffer
  runners/
    lower_trainer.py      阶段1：训下层→冻结
    upper_trainer.py      阶段2：IPPO 训上+中层
  utils/
    util.py               ★导师标准画图工具 plot_learning_curve(mean±std 阴影带)——全项目统一用它
    plotting.py           无头后端 + CSV→util 数据格式适配器（都走 util.plot_learning_curve）
    logging.py            StepLogger + InferenceTimer(单步推理耗时)
    config.py / seeding.py  YAML 加载 / 统一 set_seed(复用 util)
  results/                输出
```

---

## 运行（实现后）

```bash
python -m dhrl.run --config dhrl/configs/lower_locomotion.yaml     # 阶段1：训下层并冻结
python -m dhrl.run --config dhrl/configs/corridor_crossing.yaml    # 阶段2：IPPO
# 对照 baseline：把 config 里 algorithm 改成 mappo
```

PPO 超参（论文 §5）：Adam lr=5e-4，clip=0.2，γ=0.995。

---

## 与旧代码的关系

旧的 `dhrl_habitat/`（用了 `CentralCritic`=MAPPO/CTDE）复现成了论文的 **baseline** 而非方法。
本目录是**按论文策略（IPPO + 三层 HRL）重建的结构**；待结构确认后，把可复用的部件
（TanhNormal、截断 GAE、Habitat 环境搭建、最近邻观测、指标/出图）迁进对应模块，旧目录再清理。
