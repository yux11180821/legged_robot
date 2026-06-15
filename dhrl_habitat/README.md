# D-HRL 论文复现（Habitat 统一环境，按导师会议对齐）

复现 *Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation* (arXiv:2407.06499)，完全按 2026‑06‑12 会议纪要执行：

| 会议要求 | 落实 |
|---|---|
| 统一仿真环境（全部 Habitat） | 所有阶段共用一个 Habitat 配置（2×Spot），无第二仿真器 |
| 底层算子先单智能体训练→冻结 | `lower_train.py`（位置/速度两种模式，论文 §5.2.1 奖励）→ `load_frozen_lower` 冻结 |
| 人工指令先验证（VLM 暂缓） | `manual_validate.py`：人工指令驱动冻结算子的验证门 |
| 中层 RNN 破坏马尔可夫性 | `DHRLActor` 中的 GRU 隐状态 h_t |
| 上层=局部观测+最近邻 | `multi_env.py` 每个体只看自己+**最近**邻居（部分观测） |
| 多智能体 CTDE | 共享 actor（去中心化执行）+ `CentralCritic`（联合观测，仅训练用） |
| 2‑3 台机器人 | 2×Spot 主配置；3×Spot 实验配置 |
| 奖励曲线 ≥ 原论文 | `plot_curves.py` 论文式 shadow 曲线 + 末段均值表 |
| 每 20 step 打点 | `StepLogger(log_every=20)`，所有阶段统一 CSV |
| 打印单步推理耗时 | `InferenceTimer` 计完整层级（UL+ML+LL）前向，CUDA 同步真实计时 |

## 流水线（AutoDL）

```bash
cd $PROJECT_DIR    # /root/autodl-tmp/habitat-lab
bash run_dhrl_repro.sh preflight   # ① 必须先跑：验证新配置能加载、动作维度=3×N、标定步长
bash run_dhrl_repro.sh stage1      # ② 底层算子（默认 velocity 模式，50 万步）
bash run_dhrl_repro.sh validate    # ③ 人工指令验证门（成功率/步数/推理耗时）
bash run_dhrl_repro.sh stage2      # ④ dhrl/no_memory/no_hierarchy × 4 seeds × 200 万步
bash run_dhrl_repro.sh plots       # ⑤ 曲线 + 推理耗时表
```

环境变量可改：`SEEDS`、`ALGOS`、`N_AGENTS=3`（用 3×Spot 配置时同时设 `CONFIG=benchmark/multi_agent/hssd_spot_spot_spot.yaml`）、`LOWER_MODE=position`、`GOAL_MODE=shared|random`。

## 架构对应（论文 → 代码）

```
 e_t (相对目标 + 最近邻相对位置)          p_t (命令 + 上一步动作)
        │                                      │
   UpperLayer (MLP)                            │
        │ feature                              │
   GRUCell h_t  ← 中层时空记忆                  │
        │ command c_t ∈ (-1,1)³                │
        └──────────→ LowerOperator(冻结) ←─────┘
                          │ base velocity (vx, vy, ωz) ∈ (-1,1)³
                          ▼
            Habitat base_velocity_non_cylinder（运动学执行）
```

- **任务（走廊穿越类比）**：`goal_mode=swap` 把每台机器人的目标设为另一台的出发点，强迫它们在室内走廊/门洞处交错通过——论文走廊任务的"让行/抢行"协作在这里涌现。
- **三个对照**（论文 Fig.5/Table 1）：`dhrl`（完整方法）/ `no_memory`（去 RNN）/ `no_hierarchy`（单 MLP 直接出速度）。
- **PPO 超参取论文值**：Adam lr=5e‑4、clip=0.2、γ=0.995（λ=0.95）。
- **已修正的两个旧 bug** 默认生效：tanh 高斯动作压入 (-1,1)（带雅可比修正）；超时截断 bootstrap、成功为真终止（`compute_gae_truncated`）。

## 文件

| 文件 | 内容 | 本机已测 |
|---|---|---|
| `layers.py` | 三层网络 + 算法注册 + 冻结加载 | ✅ |
| `marl_ppo.py` | CTDE PPO 更新 + 团队优势广播 | ✅（含合成任务学习性测试） |
| `geometry.py` | 世界↔机体坐标、人工指令（推导见 docstring） | ✅（手算用例） |
| `habitat_io.py` / `lower_env.py` / `multi_env.py` | Habitat 适配（惰性导入） | 语法级 |
| `lower_train.py` / `manual_validate.py` / `upper_train.py` | 三个阶段入口 | 语法级 |
| `preflight.py` | AutoDL 首跑验证 + 步长标定 | — |
| `plot_curves.py` | 曲线与汇总表 | 语法级 |
| 配置 | `habitat-lab/.../multi_agent/hssd_spot_spot.yaml`（主）/`hssd_spot_spot_spot.yaml`（实验） | YAML 级 |

## 已知风险 / 待 AutoDL 验证

1. **新配置的 hydra 组合**：`hssd_spot_spot.yaml` 是从已知可加载的 `hssd_spot_human.yaml` 最小修改而来（PDDL 规格 `multi_agent_tidy_house` 对机器人类型通用，已查源码确认），但**必须先跑 preflight**。
2. **3×Spot**：`did_agents_collide` 硬编码 2 台（源码 multi_agent_sensors.py:45），3 台配置已剔除该度量、适配器退回距离阈值惩罚；PDDL 第三个体 grounding 未实测。
3. **步长标定**：`nominal_step_disp/yaw` 默认 10/120≈0.083（lin/ang_speed=10、habitat 默认 ctrl_freq=120Hz）；preflight 第 5 步打出实测 p95，若不符用 `lower_train.py --nominal-step-disp/--nominal-step-yaw` 覆盖。
4. **推理耗时口径**：报告时用 `--num-envs 1` 跑，计时即"一台机器人编队的单步完整层级前向"。
5. 中层 GRU 训练采用存储隐状态单步重算（无跨步 BPTT）——与旧实现一致、训练稳定；如需严格 BPTT 改序列 minibatch（论文未指明实现细节）。
