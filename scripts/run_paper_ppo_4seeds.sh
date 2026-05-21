#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-paper_ppo_pointnav_4seed_200k}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
NUM_ENVS="${NUM_ENVS:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
RNN_TYPE="${RNN_TYPE:-GRU}"
SEEDS=(${SEEDS:-100 200 300 400})

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/habitat-lab}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-habitat}"

cd "$PROJECT_DIR"
mkdir -p "logs/${EXP_NAME}" "tb/${EXP_NAME}" "data/checkpoints/${EXP_NAME}" "outputs/${EXP_NAME}"

echo "experiment=${EXP_NAME}"
echo "total_steps=${TOTAL_STEPS}"
echo "num_envs=${NUM_ENVS}"
echo "rnn_type=${RNN_TYPE}"
echo "seeds=${SEEDS[*]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="$((i % 2))"
  run_name="seed_${seed}"
  log_file="logs/${EXP_NAME}/${run_name}.log"
  tb_dir="tb/${EXP_NAME}/${run_name}"
  ckpt_dir="data/checkpoints/${EXP_NAME}/${run_name}"

  rm -rf "$tb_dir" "$ckpt_dir"
  mkdir -p "$tb_dir" "$ckpt_dir"

  echo "launch seed=${seed} gpu=${gpu} log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m habitat_baselines.run \
      --config-name=pointnav/ppo_pointnav_example.yaml \
      habitat.seed="$seed" \
      habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
      habitat_baselines.torch_gpu_id=0 \
      habitat_baselines.evaluate=False \
      habitat_baselines.total_num_steps="$TOTAL_STEPS" \
      habitat_baselines.num_environments="$NUM_ENVS" \
      habitat_baselines.rl.ddppo.rnn_type="$RNN_TYPE" \
      habitat_baselines.tensorboard_dir="$tb_dir" \
      habitat_baselines.checkpoint_folder="$ckpt_dir" \
      habitat_baselines.num_checkpoints=1 \
      habitat_baselines.log_interval="$LOG_INTERVAL"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "finished status=${status}"
exit "$status"
