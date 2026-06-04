# Habitat-Lab PPO / Zero-Shot 复现实验

这个仓库以 AutoDL 上的 `/root/autodl-tmp/habitat-lab` 工作目录为准整理，基于 Habitat-Lab / Habitat-Baselines 0.3.3。当前重点是复现论文 **Learning a Distributed Hierarchical Locomotion Controller for Embodied Cooperation** 的多智能体 D-HRL 思路，并保留 PointNav PPO / zero-shot 曲线相关脚本作为早期实验工具。

当前 GitHub 仓库只保存代码、配置、脚本和小规模测试数据。训练 checkpoint、TensorBoard、日志、视频和大型多智能体资源不放进 Git；这些文件应继续放在 AutoDL 机器的 `data/`、`logs/`、`outputs/`、`tb/`、`video_dir/` 等目录下。

## 仓库内容

- `habitat-lab/`：Habitat-Lab 源码。
- `habitat-baselines/`：PPO / DDPPO / HRL / multi-agent baseline 源码。
- `habitat-hitl/`：Habitat HITL 相关代码。
- `scripts/run_zero_shot_curve_train_4seeds.sh`：4 个随机种子并行训练 PPO 曲线。
- `scripts/run_zero_shot_eval_curve.sh`：按 checkpoint 在 held-out scene 上做 zero-shot 评估。
- `scripts/plot_zero_shot_eval_curve.py`：画 faint seed lines + mean shadow 的论文风格阴影图。
- `train_ddp.py`：Habitat multi-agent D-HRL / no-memory 双算法多 seed launcher。
- `unitree_env.py`：多智能体算法规格和 Hydra override 构造器，旧文件名仅为兼容保留。
- `unitree_nav.yaml`：D-HRL multi-agent 实验配置说明，已经不再是单智能体 Unitree 导航配置。
- `scripts/run_dhrl_multi_agent_2algorithms_4seeds.sh`：一键跑 D-HRL 和 no-memory 两个算法，各 4 个 seed。
- `scripts/plot_dhrl_multi_agent_shadow.py`：直接调用 `scripts/user_shadow_util.py::plot_learning_curve` 画两个算法的阴影图。
- `scripts/make_pointnav_heldout_dataset.py`：从 Habitat test scenes 构造 train/test held-out split。
- `scripts/run_zero_shot_replay_videos.sh`：导出 checkpoint 回放视频。
- `data/datasets/pointnav/`、`data/scene_datasets/habitat-test-scenes/`：小规模 PointNav 测试数据，方便复现实验脚本。

## 关键修改

- `habitat-baselines/habitat_baselines/rl/models/rnn_state_encoder.py`
  - 增加 `rnn_type=NONE` 的 no-memory state encoder，用于无循环记忆 baseline。
- `habitat-baselines/habitat_baselines/rl/ppo/ppo_trainer.py`
  - 兼容 PyTorch 2.6 的 checkpoint 加载。
  - checkpoint 改成固定 frame interval 保存。
  - 强制保存真正的 `final.pth`，避免最后一个 checkpoint 丢失。
- 绘图脚本支持横坐标归一化到 `[0, 1]`，并同时显示单 seed 细线、均值曲线和标准差阴影。

## D-HRL 多智能体复现

论文的核心设置是多智能体 CTDE + 分布式三层 HRL，不是单智能体 Unitree Gym wrapper。本仓库现在使用 Habitat 0.3.3 自带的 multi-agent social navigation 配置作为复现入口：

```text
social_nav/social_nav.yaml
MultiAgentAccessMgr
HRLPPO / HRLDDPPO
```

默认比较两个算法：

- `dhrl`：Distributed HRL，使用 LSTM recurrent memory。
- `no_memory`：同一套 multi-agent HRL 设置，但将 recurrent state encoder 改为 `rnn_type=NONE`，对应论文 Fig. 5 / Table 1 的 no spatiotemporal memory 消融。

AutoDL 上运行：

```bash
cd /root/autodl-tmp/habitat-lab

# 国内网络如果无法访问 HuggingFace，先配置镜像：
git config --global url."https://hf-mirror.com/".insteadOf "https://huggingface.co/"

# 下载 Habitat-3 social-nav 所需资源。已有文件不会覆盖。
python -m habitat_sim.utils.datasets_download --uids \
  hssd-hab hab3-episodes habitat_humanoids hab3_bench_assets hab_spot_arm ycb \
  --data-path data --no-replace

# 检查资源是否齐全。
python scripts/check_hab3_social_nav_assets.py --project-dir /root/autodl-tmp/habitat-lab

# 如果镜像仍卡在 cas-bridge.xethub，说明当前节点解析不到 HuggingFace/Xet
# 的对象存储域名。不要直接开训，先从另一台机器或已有缓存拷贝完整 data。

EXP_NAME=dhrl_social_nav_2algorithms \
TOTAL_STEPS=2000000 \
NUM_ENVS=8 \
CKPT_INTERVAL_FRAMES=100000 \
SEEDS="100 200 300 400" \
GPUS="0" \
MAX_PARALLEL=1 \
bash scripts/run_dhrl_multi_agent_2algorithms_4seeds.sh
```

脚本会生成：

- `tb/dhrl_social_nav_2algorithms/dhrl/seed_*`
- `tb/dhrl_social_nav_2algorithms/no_memory/seed_*`
- `data/checkpoints/dhrl_social_nav_2algorithms/{dhrl,no_memory}/seed_*`
- `results/dhrl_social_nav_2algorithms/dhrl_multi_agent_reward_shadow.png`

单独重画阴影图：

```bash
python scripts/plot_dhrl_multi_agent_shadow.py \
  --tb-root tb/dhrl_social_nav_2algorithms \
  --out-dir results/dhrl_social_nav_2algorithms \
  --metric reward \
  --bin-size 100000
```

这个绘图脚本会把 TensorBoard 标量转成 `scripts/user_shadow_util.py` 需要的 `{steps, reward, seed}` 格式，并直接调用你给的 `plot_learning_curve` 画两个算法的均值/标准差阴影图。

## AutoDL 运行环境

推荐在 AutoDL 上使用已有环境：

```bash
cd /root/autodl-tmp/habitat-lab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate habitat
python -c "import habitat, habitat_baselines; print(habitat.__version__, habitat_baselines.__version__)"
```

当前实验按 Habitat 0.3.3 整理。大型资源不随 GitHub 同步，需要在 AutoDL 本地准备，例如 humanoid / robot / object 资产、预训练策略、训练 checkpoint 等。

## 训练 4 个随机种子

默认脚本会跑 4 个 seed：`100 200 300 400`。可以通过环境变量改训练步数、checkpoint 间隔和实验名。

```bash
cd /root/autodl-tmp/habitat-lab

EXP_NAME=rlvigen_pointnav_zeroshot_curve_1m_dense20k \
TOTAL_STEPS=1000000 \
CKPT_INTERVAL_FRAMES=20000 \
NUM_ENVS=4 \
SEEDS="100 200 300 400" \
bash scripts/run_zero_shot_curve_train_4seeds.sh
```

输出目录：

- checkpoint：`data/checkpoints/${EXP_NAME}/seed_${SEED}/`
- 日志：`logs/${EXP_NAME}/`
- TensorBoard：`tb/${EXP_NAME}/`

## Zero-Shot 评估

训练完成后，对每个 checkpoint 在 held-out test split 上评估：

```bash
cd /root/autodl-tmp/habitat-lab

EXP_NAME=rlvigen_pointnav_zeroshot_curve_1m_dense20k \
EVAL_NAME=rlvigen_pointnav_zeroshot_curve_1m_dense20k_eval_vangogh_test_150ep \
TEST_EPISODES=150 \
EVAL_SPLIT=test \
SEEDS="100 200 300 400" \
bash scripts/run_zero_shot_eval_curve.sh
```

评估日志会写到：

```text
logs/${EVAL_NAME}/
```

## 绘制论文风格阴影图

横坐标可使用真实 frame，也可以归一化到 `[0, 1]`。论文图里常见的是 normalized training steps：

```bash
cd /root/autodl-tmp/habitat-lab

python scripts/plot_zero_shot_eval_curve.py \
  --log-dir logs/rlvigen_pointnav_zeroshot_curve_1m_dense20k_eval_vangogh_test_150ep \
  --checkpoint-root data/checkpoints/rlvigen_pointnav_zeroshot_curve_1m_dense20k \
  --out-dir results/rlvigen_pointnav_zeroshot_curve_1m_dense20k_eval_vangogh_test_150ep_paper \
  --seeds 100 200 300 400 \
  --x-axis normalized \
  --total-steps 1000000
```

输出包括：

- `zero_shot_eval_shadow_curve.png`
- `zero_shot_eval_raw.csv`
- `zero_shot_eval_summary.csv`

## 视频回放

如果需要看训练后策略的视频，使用：

```bash
cd /root/autodl-tmp/habitat-lab

EXP_NAME=rlvigen_pointnav_zeroshot_curve_1m_dense20k \
SEEDS="100 200 300 400" \
bash scripts/run_zero_shot_replay_videos.sh
```

视频输出在：

```text
video_dir/
```

## 不进入 Git 的文件

以下文件或目录体积大，默认不提交：

- `data/checkpoints/`
- `data/humanoids/`
- `data/objects/`
- `data/robots/`
- `logs/`
- `outputs/`
- `tb/`
- `video_dir/`
- `*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.mp4`

如需同步大型资源，建议使用 AutoDL 云盘、对象存储或 Git LFS，不建议直接塞进普通 Git 仓库。
