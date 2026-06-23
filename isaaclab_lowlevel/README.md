# 下层 per-leg MARL 骨架（Isaac Lab）

混合架构的**下层**：把一台四足拆成 **FL / FR / RL / RR 四个 Executor**，用 **MAPPO（CTDE，中心化 critic）** 学全身运动；上层 Commander（在 Habitat 里）以后通过速度命令 `(vx, vy, ωz)` 驱动这个冻结的下层。

> 这部分**必须**在物理运动仿真器里跑（Habitat 做不了关节级/接触力/平衡）。本骨架基于 **Isaac Lab v2.x**（`isaaclab.*`），机器人默认 **Unitree Go2**。

## 它如何对应你的方法

| 你的方法 | 本骨架的实现 |
|---|---|
| 一台机器人 → 多智能体 | `possible_agents = [FL, FR, RL, RR]`，每腿一个 agent |
| 下层 Executor：每个控制一条腿 | 每 agent 动作 = 该腿 3 个关节的位置目标（叠加在默认站姿上，由 PD 执行器出力矩） |
| 仅本体感知、看不到全局地图 | 每 agent 观测 = 自身腿关节位/速 + 足端接触力 + 基座 IMU/姿态/角速度 + 速度命令；**无视觉、无地图** |
| MAPPO 协作 | skrl MAPPO：actor 输入 `OBSERVATIONS`（本腿），critic 输入 `STATES`（全身状态）→ 真正的中心化 critic |
| 姿态平衡=共享奖励 | `_get_rewards` 里的 **shared** 项（速度跟踪 + 平腰/基座高度/角速度惩罚 + 存活），同一标量广播给 4 条腿 |
| 各腿最小化自身能耗 | 每 agent 额外加 **per-leg** 项：`Σ|τ·q̇|`（机械功率/能量）+ `Στ²` + 关节加速度 + 动作平滑 |
| 上层 Commander 接口 | `env.set_velocity_commands(cmd)` + `cfg.external_commands=True`，由上层注入 `(vx,vy,ωz)` |

核心文件 [`quadruped_four_legs_env.py`](quadruped_four_legs_env.py)；MAPPO 配置 [`agents/skrl_mappo_cfg.yaml`](agents/skrl_mappo_cfg.yaml)。

## 安装与运行（AutoDL / GPU 机器）

需要 **Isaac Sim + Isaac Lab（含 `isaaclab_assets`、`isaaclab_rl`）+ skrl 2.x**。装好 Isaac Lab 后，从本仓库根目录：

```bash
# 训练（MAPPO，中心化 critic）
./isaaclab.sh -p isaaclab_lowlevel/train.py --algorithm MAPPO --num_envs 4096 --headless --max_iterations 1500

# 消融（IPPO，去中心化 critic —— 仅 critic 输入从 STATES 换成 OBSERVATIONS）
./isaaclab.sh -p isaaclab_lowlevel/train.py --algorithm IPPO --num_envs 4096 --headless --max_iterations 1500

# 回放/评估 + 用外部速度命令驱动（模拟上层 Commander）
./isaaclab.sh -p isaaclab_lowlevel/play.py --algorithm MAPPO --checkpoint <agent.pt> --num_envs 16 --external_commands
```

`./isaaclab.sh -p` 用的是 Isaac Lab 自带的 python。确保本仓库在 `PYTHONPATH`（`train.py` 会自动把仓库根加进去）。

## 设计要点（已按官方源码核对）

- **关节/足端索引一律运行时解析**（`find_joints` / `find_bodies`）：Isaac Lab 的 DOF 顺序由 USD 决定，不是 leg 顺序，硬编码索引是最常见的隐性 bug。
- **中心化 critic**：`state_space` 设正整数并实现 `_get_states()`（全身状态），而非 `-1`（仅拼接各腿观测）——这样 critic 才看得到真正的全局状态（基座速度、接触力、命令）。
- **能量用 `applied_torque`**（经 DCMotor 限幅后的实际力矩），不是原始动作。
- skrl 配置里 `time_limit_bootstrap: True`：超时截断时对价值做 bootstrap（避免把截断当真终止——这正是你旧 Habitat 代码里的那个 GAE bug，这里默认修好）。

## 扩展点

- **换成轮足机器人**：把 `cfg.robot_cfg` 换成你的 USD/`ArticulationCfg`，并改 `cfg.leg_joint_names`（每条腿加上轮关节）；同步把 `JOINTS_PER_LEG`（及 `PER_LEG_OBS_DIM`/`STATE_DIM`）调整为新关节数。若各腿关节数不同，需要给每个 agent 单独的 model 规格（skrl 支持按 agent id 分别配置）。
- **足端接触力维度**：当前每腿用接触力**幅值（1 维）**；要 3 维力或足端空中时间（`current_air_time`/`last_air_time`，已开 `track_air_time`），在 `_get_observations` 里加并同步改 `PER_LEG_OBS_DIM`。
- **VLM 离线蒸馏**：那是**上层 Commander** 的事（在 Habitat 那侧），与本下层解耦；这里只暴露 `set_velocity_commands` 接口。
- **冻结导出给上层**：训完用 skrl checkpoint；在上层训练里加载该 actor、`set_running_mode("eval")`、`torch.inference_mode()`，把 `(vx,vy,ωz)` 喂进来即可（见 `play.py`）。

## 注意 / 当前限制

- **平地假设**：基座高度/姿态奖励按平地写；要上崎岖地形需加 height-scan 传感器与地形 cfg。
- **版本**：按 Isaac Lab **v2.x**（`isaaclab.*`、`isaaclab_assets`、`isaaclab_rl`）写。若你是 v1.4.1 及更早，把所有 `isaaclab` 前缀换成 `omni.isaac.lab`、`isaaclab_assets`→`omni.isaac.lab_assets`，skrl 包装器路径相应调整。
- **未调参**：奖励权重/层宽/`tracking_sigma` 是合理起点，需在你机器上按收敛情况调。
- 本机无 GPU/Isaac Lab，代码做了静态核对（API、维度、索引解析），**端到端训练请在你的 AutoDL 上验证**。
```
