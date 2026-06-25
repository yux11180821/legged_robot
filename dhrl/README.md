# `dhrl/` — 分布式分层运动控制（arXiv:2407.06499 复现）

复现 **Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation**
（Hong, Huang, Liu — Tsinghua, CoRL 2024）的**策略本身**。代码按标准 on-policy MARL
框架组织（参考 MARL_for_MeltingPot：`algorithms / networks / envs / memories / runners / utils / configs`），
每个 class/函数/方法都有详尽 docstring（参数形状 + 返回 + 对应论文章节）。

---

## ⚠️ 复现的算法是 IPPO（去中心化），不是 MAPPO/CTDE

重读论文最关键的一点（也是之前代码的核心问题）：

- §5：“**Independent PPO (IPPO)** in a multi-agent setting is employed as the algorithm.”
- §4.2：完全去中心化、部分观测，“**unattainable by CTDE frameworks**”。
- Fig.5(b)(c)：`Distributed HRL (IPPO)` vs `Centralized Training (MAPPO)` → **MAPPO 收敛更慢、agent 多了不收敛**。

**结论：MAPPO/中心化 critic 是论文要打败的 baseline，不是论文方法。**
本包以 **IPPO 为主**（`algorithms/ippo.py`），MAPPO 仅作对照 baseline（`algorithms/mappo.py`）。

---

## 整体架构：每个 agent 一份三层 HRL（论文 Fig.2）

```
                         ┌──── 每个 agent 各跑一份（同构共享参数，完全去中心化）────┐
  外部观测 e_t ─────────▶│  UL 上层(感知) ──feature──▶ ML 中层(RNN, 记忆 h_t)      │
  [环境 + 仅最近邻]      │                                   │ command(命令)        │
                         │                                   ▼                      │
  本体观测 p_t ─────────────────────────────────────▶ LL 下层(冻结运动算子) ──▶ a_t │
  [位置/速度/关节力矩]   └──────────────────────────────────────────────────────────┘
```

| 层 | 文件 | 职责（论文） |
|---|---|---|
| **UL 上层** | `networks/upper_layer.py` | 感知 e_t → 特征。e_t = 环境 + **仅最近邻**相对位置（§4.2, Fig.3，部分观测=可扩展关键）|
| **ML 中层** | `networks/middle_layer.py` | **RNN** 维护时空记忆 h_t → 输出**命令**（§4.3；无 RNN 消融惨败，Table 1）|
| **LL 下层** | `networks/lower_layer.py` | **预训练后冻结**的运动算子：p_t + 命令 → a_t；位置/速度两模式（§4.1, §5.2.1）|
| 拼装 | `networks/hrl_policy.py` | UL→ML→冻结LL；PPO 优化的动作是**命令**（UL+ML 训、LL 冻结）|

---

## 两阶段训练流程（§5.2）

```
阶段1 (单智能体)                       阶段2 (多智能体, 去中心化 IPPO)
┌──────────────────────────┐         ┌────────────────────────────────────────┐
│ ppo_single 训 LL 下层    │  冻结   │ IPPO 训 UL + ML(RNN)，LL 保持冻结        │
│ 位置: r=1/L2距离         │ ──────▶ │ 在 协同搬运/走廊穿越/峡谷架桥 上训练     │
│ 速度: r=速度点积         │ (位置)  │ 每个 agent 只看自己的观测，绝不拼全局    │
└──────────────────────────┘         └────────────────────────────────────────┘
  runners/lower_trainer.py             runners/upper_trainer.py
```

同构共享一套策略 + 共享奖励（Eq.1）；异构（峡谷架桥）用按物种乘积目标（Eq.2）。
GAE 截断感知：超时 bootstrap V(终态)，真终止不 bootstrap。

---

## 三个任务（§5.1, Fig.1）

| 任务 | 文件 | 说明 | 论文成功率 |
|---|---|---|---|
| Cooperative Transport | `envs/cooperative_transport.py` | 同构群体协同搬物到目标区 | 76.8% |
| Corridor Crossing | `envs/corridor_crossing.py` | 同构，窄走廊轮流通过（让行） | 88.4% |
| Ravine Bridging | `envs/ravine_bridging.py` | **异构**（2 套策略），一组推桥让另一组过 | 50.4% |

（消融 Table 1：No Hierarchy 全 0%；No Spatiotemporal Memory ~5–12%。）

---

## 目录结构

```
dhrl/
  run.py                  统一入口（清晰调用链：load_config→set_seed→stage 分派）
  configs/                YAML 配置（base + 阶段1 + 三个任务）
  algorithms/
    ippo.py               ★ 论文方法：Independent PPO（去中心化，去中心 critic）
    mappo.py              CTDE baseline（仅对照，论文要打败的对象）
    ppo_single.py         阶段1 单智能体 PPO（训下层算子）
  networks/
    upper_layer.py        UL 感知
    middle_layer.py       ML 时空记忆 RNN（use_memory=False = 无记忆消融）
    lower_layer.py        LL 冻结运动算子（阶段1可训 + freeze/load_frozen）
    hrl_policy.py         三层拼成一个 agent 的策略
    critic.py             去中心 critic(IPPO) + 中心 critic(MAPPO 基线)
    distributions.py      TanhNormal 动作分布（含 deterministic 供 eval 锁网络）
  envs/
    base.py               多智能体 env / Backend 接口
    obs.py                §4.2/Fig.3 仅最近邻的部分观测（可扩展关键）
    habitat_backend.py    Habitat 仿真后端（论文用 IsaacSim+Ant；我们 Habitat+Spot）
    cooperative_transport.py / corridor_crossing.py / ravine_bridging.py / locomotion.py
  memories/
    rollout_buffer.py     带 RNN 隐状态 + 截断 GAE 的 on-policy buffer
  runners/
    lower_trainer.py      阶段1：训下层 → 冻结位置模式
    upper_trainer.py      阶段2：IPPO 训 UL+ML（冻结 LL 之上）
  utils/
    util.py               ★导师标准画图 plot_learning_curve（全项目统一用它，原样保留）
    plotting.py           无头后端 + CSV→util 适配器
    logging.py            StepLogger + InferenceTimer(单步推理耗时)
    config.py / seeding.py  YAML 加载 / set_seed
  results/                输出
```

---

## 运行（从 `dhrl/` 目录内运行，与 MARL_for_MeltingPot 一致）

```bash
cd dhrl
python run.py --config configs/lower_locomotion.yaml      # 阶段1：训下层并冻结
python run.py --config configs/corridor_crossing.yaml     # 阶段2：IPPO
# 对照 baseline：把 config 里 algo 改成 algorithms.mappo.MAPPO
```

PPO 超参（论文 §5）：Adam lr=5e-4，clip=0.2，γ=0.995。

**代码风格对齐 MARL_for_MeltingPot**：
- 算法是**一个 agent 类**（`IPPO(env_info, args)`，内部建 actor+critic+优化器，含 `init_hidden/take_action/update/save`，GAE 和更新都在内部）；
- 训练是**一个 `train(args, envs, agent, buffer)` 大函数**（循环内联、从上读到下，无工厂 lambda / 无 hack）；
- **普通类，无 ABC/Protocol**；env 有 `get_env_info()`；
- **扁平 args 配置**（`get_config()` 把 YAML 读成 namespace，`load_algorithm` 按字符串加载算法类）；
- 调用链：`run.main()` → `get_config()` + `set_seed` → 按 `stage` 调 `train_lower(args)` 或 建 env/agent/buffer 后调 `train(args, envs, agent, buffer)`。

> 现状：全部代码已实现且每个函数有注释，非 Habitat 部分（网络/IPPO/GAE/最近邻观测/两阶段循环）
> 本地全程跑通；Habitat 后端懒加载，待上服务器接真环境。
