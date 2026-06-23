# Go2 per-leg MARL 下层（MuJoCo）

把论文"分布式分层 MARL"的思路落到**一台四足的 4 条腿**上——**每条腿 = 一个智能体**（FL/FR/RL/RR）。这是你迁移到轮足机器人的**下层运动算子**：先用 Go2 在 MuJoCo 跑通，再换成你的轮足 USD。

## 对应你的方法（论文下层）
| 你的设计 | 实现 |
|---|---|
| 一台机器人→4 腿智能体 | `Go2LegMARLEnv`，每 agent 控自己腿的 3 关节（位置目标→Python PD→力矩） |
| 仅本体感知、无视觉 | 每 agent 观测=本腿关节位/速+足端接触+基座 IMU(线/角速度+投影重力)+速度命令（22 维） |
| 姿态平衡=共享奖励 | `_rewards_dones` 的 shared 项（速度跟踪+平腰+基座高度+存活）广播给 4 腿 |
| 各腿最小化自身能耗 | 每腿 per-leg 项：`Στ²`（用 `actuator_force`）+ 动作平滑 |
| CTDE | 共享 `LegActor`（去中心化执行）+ `CentralCritic`（全身 52 维 state，仅训练用） |
| 上层接口 | `env.set_velocity_commands((vx,vy,ωz))` + `cfg.external_commands=True`（以后由 Habitat Commander 驱动这个冻结下层） |

复用了项目里**已修 bug 的 CTDE‑PPO 内核**（截断感知 GAE + **LR 退火 + KL 早停**，正是上一轮 dhrl 崩溃后加的稳定器）、tanh 动作分布、指标采集——只有 MuJoCo 环境和 per-leg actor 是新写的。

## 在 AutoDL 上跑（已装 mujoco 3.3.7 + Go2 模型）
```bash
cd /root/autodl-tmp        # mujoco_menagerie/unitree_go2/scene.xml 在这里
source /root/miniconda3/etc/profile.d/conda.sh && conda activate habitat
# 纯逻辑自测（秒级）
python /root/autodl-tmp/habitat-lab/mujoco_leg_marl/selftest.py
# 训练（headless，mj_step 不需要显示）
PROJECT_DIR=/root/autodl-tmp/habitat-lab \
python /root/autodl-tmp/habitat-lab/mujoco_leg_marl/train.py \
  --model-path /root/autodl-tmp/mujoco_menagerie/unitree_go2/scene.xml \
  --num-envs 32 --total-steps 3000000 --seed 100
```
指标 CSV：`results/go2_leg_marl/legmarl_seed_100_metrics.csv`（每 2000 步一点 + 单步推理耗时）。曲线：`reward`（学会"按命令速度走且不摔"）、`ep_len`（站住多久）应往上走。

## 已核对的 Go2 模型事实
12 关节 `<leg>_<hip|thigh|calf>_joint`；作动器是**力矩电机**（ctrlrange=力矩上限 23.7/23.7/45.43）→ 我加 Python PD（kp=25, kd=0.5, action_scale=0.25）；足端碰撞几何 `FL/FR/RL/RR`（calf 上）；地面 `floor`；基座 `base`；home 站姿 `[hip0,thigh0.9,calf-1.8]`、高 0.27m；dt=0.002(500Hz)、decimation 10→50Hz 控制；**无传感器**（IMU/接触都从 qpos/qvel/contact 算）；`mj_step` 纯物理、**headless 无需 GL**。

## 换成你的轮足机器人
1. 准备轮足 USD/MJCF（MuJoCo 用 MJCF/XML；URDF 可转）。
2. `Go2LegMARLCfg.model_path` 指向它；改 `LEG_NAMES` 对应的关节/作动器/足端几何名（每条腿若多一个**轮关节**，把 `JOINTS_PER_LEG` 从 3 改成 4 并同步 `PER_LEG_OBS_DIM`/`STATE_DIM`）。
3. 轮子通常是**速度控制**而非位置 PD——给 wheel 关节单独走速度命令（在 `_apply_pd_and_step` 里分支）。

## 注意
- 纯 CPU `mj_step`（plain mujoco，非 MJX），N 个并行 MjData 在 Python 循环里步进；num-envs 32 足够"跑通"，要更快可上 MJX(JAX)。
- 奖励权重/PD 增益/命令范围是合理起点，按收敛调（`base_height_target=0.30`、`tracking_sigma=0.25` 等）。
- 纯核心（policy/CTDE 更新/GAE/指标）已本机自测；MuJoCo 端到端在 AutoDL 验证。
