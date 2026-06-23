# 在 Habitat 复现 D-HRL（分层多智能体协同运动）

复现论文 **Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation**（arXiv:2407.06499）的分层思路。

设计要点：**全程在 Habitat（ReplicaCAD，2 台 Spot）完成，不脱离平台，下层不使用分布式**。基于 Habitat-Lab / Habitat-Sim 0.3.3。复现代码在 [`dhrl_habitat/`](dhrl_habitat)，底层修过 bug 的 PPO 核心在 [`habitat_commander/`](habitat_commander)。checkpoint / 大型数据 / 视频不进 Git。

## 两阶段复现流程

**阶段 1 — 下层：单智能体 PPO 点到点导航（训完冻结）**
用单智能体 PPO 输出**一个 Spot** 的 base 速度（holonomic），学习导航到目标点；**同一时间只训这一个 Spot**，其余 Spot 用**规则**（NavMesh 航点 → 速度）走到各自目标，充当移动障碍。
奖励 = 测地距离进度 + 到达奖励 − 碰撞 − 邻近 − 时间惩罚。到达 = 真终止，超时 = 截断（bootstrap V(终态)）。训完冻结。

**阶段 2 — 上层：MAPPO 输出"给下层的命令"**
在**冻结**的下层之上，用多智能体 MAPPO（共享 actor + 中心 critic，CTDE）训练协调规划。
上层命令是一个**虚拟目标点（ego 方向"胡萝卜"）**，喂给冻结下层 → base 速度。任务 = swap 走廊互让（每台机器人去对方起点，强制穿越，涌现避让/绕行）。

```
观测 e_t → 上层 MAPPO actor → 命令(虚拟 goal_ego) ┐
                                                   ├→ 冻结下层导航策略 → base 速度 → Habitat
                          当前邻居相对位置 ─────────┘
```

## 关键文件

`dhrl_habitat/`
- `lower_nav_env.py` — 下层环境适配器（agent_0 学习导航，agent_1 脚本走）
- `lower_nav_train.py` — 下层单智能体 PPO 训练
- `upper_mappo_train.py` — 上层 MAPPO 训练（加载冻结下层）
- `multi_env.py` — 多机器人适配器（swap / shared / random 目标模式，团队奖励）
- `marl_ppo.py` — MAPPO 核心（共享 actor + 中心 critic + 截断 GAE + KL 早停）
- `layers.py` — DHRLActor / FlatActor / CentralCritic / 下层算子，算法注册表
- `geometry.py` · `habitat_io.py` — ego 变换、测地距离、NavMesh 航点
- `make_replica_cad_episodes.py` — ReplicaCAD 导航 episode 生成
- `plot_curves.py` — 论文风格阴影曲线 + 单步推理耗时表

`habitat_commander/` — 修过 bug 的单智能体 PPO 核心（TanhNormal 动作落在 (-1,1)、截断感知 GAE、指标采集），下层复用。

配置：`habitat-lab/habitat/config/benchmark/multi_agent/replica_cad_spot_spot.yaml`（2×Spot，`base_velocity_non_cylinder`，`RearrangeEmptyTask`）。

## 运行

环境（已在 AutoDL 的 `habitat` conda 环境验证；需要 Habitat 0.3.3 + ReplicaCAD/Spot 资产 + 仓库内的 habitat_simulator 兼容补丁）：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate habitat
export PROJECT_DIR=$(pwd)
# 生成导航 episodes（产出 data/datasets/replica_cad_nav/{split}/nav_episodes.json.gz）
python dhrl_habitat/make_replica_cad_episodes.py
```

**阶段 1 — 训下层（单 seed）：**

```bash
python dhrl_habitat/lower_nav_train.py --num-envs 4 --total-steps 300000 --seed 100
# 冒烟自检： python dhrl_habitat/lower_nav_train.py --smoke
# 产物： results/lower_nav/lower_nav_seed_100_metrics.csv
#        data/checkpoints/lower_nav/lower_nav_seed_100_final.pt   ← 冻结下层
```

**阶段 2 — 训上层 MAPPO（加载冻结下层）：**

```bash
python dhrl_habitat/upper_mappo_train.py --algorithm dhrl --seed 100 \
  --lower-ckpt data/checkpoints/lower_nav/lower_nav_seed_100_final.pt \
  --num-envs 4 --total-steps 500000 --goal-mode swap
# 消融（论文 Fig.5 / Table 1）： --algorithm no_memory | no_hierarchy
# 产物： results/upper_mappo/dhrl_seed_100_metrics.csv
```

**4-seed 一键流水线 + 阴影图：**

```bash
bash scripts/run_4seed_pipeline.sh   # 4 seed：下层 → 冻结 → 上层 → 出图
bash scripts/run_4seed_upper.sh      # 只重跑上层（复用已冻结的 4 个下层）
```

**单独出阴影曲线 + 推理耗时表：**

```bash
python dhrl_habitat/plot_curves.py --results-dir results/upper_mappo \
  --algorithms dhrl --metric success_rate --raw-x
python dhrl_habitat/plot_curves.py --results-dir results/lower_nav \
  --algorithms lower_nav --metric reward --raw-x
```

## 结果（ReplicaCAD，2 Spot）

- **下层**：导航成功率 ~0.74（收敛冻结），单步推理 ~1.0 ms。
- **上层 MAPPO（swap）**：成功率峰值 ~0.66，多 seed 均值 ~0.5；全层级单步推理 ~1.0 ms（对比慢速 VLM 控制器的效率卖点）。
- 曲线在 `results/lower_nav/*.png`、`results/upper_mappo/*.png`。

## 下一步

`mujoco_leg_marl/` — 把这套分层 MARL 迁移到**一台轮足机器人的四条腿**（4 条腿 = 4 个智能体，per-leg 分布式）。这是我们自己的方法、下一步的工作（尚未完成）。

## 不进 Git 的文件

checkpoint、大型数据/资产、日志、TensorBoard、视频均不提交（见 `.gitignore`：`data/checkpoints/`、`logs/`、`tb/`、`*.pt` 等），放在训练机本地。

> 仓库还保留少量早期 PointNav / social-nav 实验脚本（`train_ddp.py`、`unitree_env.py`、`scripts/run_zero_shot_*`、`scripts/plot_zero_shot_*` 等）作为历史工具，与上面的复现流程相互独立。
