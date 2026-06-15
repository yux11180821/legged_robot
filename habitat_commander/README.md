# Habitat 复现核心（按会议纪要对齐）

> **会议修正**：① 统一仿真环境=**全部在 Habitat**（IsaacLab 双仿真路线已搁置）；② **多机器人 2‑3 台协同**（非 per‑leg）；③ 指令先用**人工指令**验证，**VLM 路线暂缓**（执行随机误差累积、推理慢）；④ 硬指标：奖励曲线 ≥ 原论文、打印**单步推理耗时**、每 **20 step** 打点。

这个目录是**与拆法无关的第 1 步基建**：单机器人算子（PPO）+ 指标采集 + 人工指令基线，之后供"多机器人分层 CTDE"复用。命令是 holonomic `(vx, vy, ωz)`，训练时 Habitat 运动学 `base_velocity` 控制器充当"理想冻结下层"。

## 对应你的方法

| 你的方法（上层 Commander） | 实现 |
|---|---|
| 全局导航，输入深度/LiDAR | [`CommanderObs`](habitat_env.py) 把深度+目标罗盘拼成观测；感知编码器在 [`commander_policy.py`](commander_policy.py)（含可选 `DepthCNN`） |
| 输出 `(vx, vy, ωz)` | `CommanderPolicy` → **Tanh 压缩高斯**，输出落在 `(-1,1)³`，正好是 Habitat `base_velocity` 期望区间 |
| 人工指令 →（可选 BC 暖启动）→ PPO | [`vlm_distill.py`](vlm_distill.py)：`ManualCommandPolicy`（人工指令）+ `behavior_clone`；`VLMTeacher` 仅留作未来钩子 |
| 端到端 PPO | [`ppo.py`](ppo.py)：**截断感知 GAE** + clip PPO |
| 每 20 step 打点 + 单步推理耗时 | [`metrics.py`](metrics.py)：`StepLogger`（按 20 step 分桶写 CSV）+ `InferenceTimer`（CUDA 同步计时） |

## 已修复你旧代码的两个 bug（已用 `selftest.py` 数值验证）

1. **动作缩放**：旧代码动作空间 `[-20,20]` 但 env 内部 clip 到 `[-1,1]`→ 95% 范围饱和。现在用 **Tanh 压缩高斯**，输出 `(-1,1)` 且 log‑prob 含 tanh 雅可比修正，策略梯度与实际执行动作一致。
2. **截断当真终止**：旧 GAE 把超时 `done` 当真终止、丢掉 bootstrap。现在 `compute_gae_truncated` 对**超时截断**做 `V(terminal)` bootstrap、对**任务真终止**不 bootstrap，且不跨 episode 泄漏优势。

```
$ python habitat_commander/selftest.py
[ok] GAE truncation: terminal adv=0.0000  truncated adv=4.9500 (= gamma*5)
[ok] GAE no cross-episode leak: adv[t0]=1.9405 (= 1+gamma*lam), adv[t1]=1.0000
[ok] tanh policy: env_action in (-0.878,0.674); log_prob consistent (max|delta|=0.00e+00)
ALL SELF-TESTS PASSED
```

## 文件

| 文件 | 内容 | 依赖 |
|---|---|---|
| [commander_policy.py](commander_policy.py) | `CommanderPolicy`（感知+GRU+Tanh 高斯）、`DepthCNN` | 纯 torch（可测） |
| [ppo.py](ppo.py) | 截断感知 GAE、PPO 更新 | 纯 numpy/torch（可测） |
| [vlm_distill.py](vlm_distill.py) | `VLMTeacher`/`OracleGoalTeacher`/`behavior_clone` | 纯（可测） |
| [habitat_env.py](habitat_env.py) | Habitat 适配：建环境、观测、导航奖励 | 惰性导入 habitat |
| [train.py](train.py) | 训练：可选 BC 暖启动 → PPO | habitat |
| [selftest.py](selftest.py) | 纯逻辑自测（已通过） | numpy/torch |

## 运行（AutoDL，需 habitat env + 数据）

默认 `--config-name` **复用你已能加载的 social‑nav 配置**，只控制 `agent_0` 的 base 速度、并把它强制成 holonomic（`enable_lateral_move=True`），导航奖励在训练器内算（不依赖脆弱的 reward‑measure 配置）：

```bash
# VLM 蒸馏暖启动 + PPO
python habitat_commander/train.py --bc-warmstart --total-steps 2000000

# 纯 PPO（无暖启动）
python habitat_commander/train.py --total-steps 2000000 --num-envs 4
```

关键参数（按你的 episode/传感器调整）：
- `--action-key`（默认 `agent_0_base_velocity`）：被控的 holonomic base 速度动作。
- `--depth-key`（默认 `agent_0_spot_head_stereo_depth_sensor`）：感知输入。
- `--goal-key`（默认 `agent_0_goal_to_agent_gps_compass`）：一个 `(rho, phi)` 罗盘传感器；导航奖励用它的 `rho`。

## 接入真实 VLM（替换 OracleGoalTeacher）

`OracleGoalTeacher` 是零依赖占位（用目标罗盘指向目标），让 BC 流程先跑通。换成真 VLM：实现 `VLMTeacher` 协议——把深度/RGB 帧渲染出来，prompt 一个 VLM（Claude / GPT‑4V / Qwen‑VL）给出朝目标的航向，解析成 `(vx, vy, ωz) ∈ [-1,1]`。`train.py` 里把 `teacher = OracleGoalTeacher()` 换成你的实现即可。

## 与下层闭环

1. 在 IsaacLab 训好 per‑leg 下层并冻结（导出 checkpoint）。
2. Commander 用 Habitat 的运动学速度当理想下层训练（本目录）。
3. 闭环评估：把 Commander 输出的 `(vx,vy,ωz)` 通过 `env.set_velocity_commands(...)` 喂给 IsaacLab 下层（见 `../isaaclab_lowlevel/play.py`）。

## 注意 / 限制

- **导航 episode/目标**：`--goal-key` 要对应你 episode 实际提供的 `(rho,phi)` 罗盘传感器（social‑nav 里是 `agent_0_goal_to_agent_gps_compass`，目标是人；做点到点导航需要带导航目标的 episode）。
- **感知**：默认把深度降采样成扁平向量喂 MLP；要更强感知把 `DepthCNN` 接进策略（改成 dict 观测）。
- **GRU 训练**：rollout 存 hidden 单步重算（无 BPTT），对上层规划够用；要严格时序可改成序列 minibatch。
- 纯核心（policy/ppo/vlm）已在本机用 `selftest.py` 跑通；**Habitat 端到端训练请在 AutoDL 验证**（本机无 habitat/数据）。
