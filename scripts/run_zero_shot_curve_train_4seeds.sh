#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-rlvigen_pointnav_zeroshot_curve_1m}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-4}"
CKPT_INTERVAL_FRAMES="${CKPT_INTERVAL_FRAMES:-100000}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
if [ -z "${DATA_PATH+x}" ]; then
  DATA_PATH='data/datasets/pointnav/heldout-vangogh/v1/{split}/{split}.json.gz'
fi
SEEDS=(${SEEDS:-100 200 300 400})

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/habitat-lab}"
cd "$PROJECT_DIR"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate habitat

mkdir -p "logs/${EXP_NAME}" "tb/${EXP_NAME}" "data/checkpoints/${EXP_NAME}" "outputs/${EXP_NAME}"

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

echo "experiment=${EXP_NAME}"
echo "total_steps=${TOTAL_STEPS}"
echo "ckpt_interval_frames=${CKPT_INTERVAL_FRAMES}"
echo "data_path=${DATA_PATH}"
echo "seeds=${SEEDS[*]}"

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

  echo "launch seed=${seed} gpu=${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export HABITAT_CKPT_INTERVAL_FRAMES="$CKPT_INTERVAL_FRAMES"
    python -m habitat_baselines.run \
      --config-name=pointnav/ppo_pointnav_example.yaml \
      habitat.seed="$seed" \
      habitat.dataset.split=train \
      "habitat.dataset.data_path='${DATA_PATH}'" \
      habitat_baselines.evaluate=False \
      habitat_baselines.total_num_steps="$TOTAL_STEPS" \
      habitat_baselines.num_environments="$NUM_ENVS" \
      habitat_baselines.tensorboard_dir="$tb_dir" \
      habitat_baselines.checkpoint_folder="$ckpt_dir" \
      habitat_baselines.num_checkpoints="$NUM_CHECKPOINTS" \
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
