#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-rlvigen_pointnav_zeroshot_curve_200k}"
VIDEO_NAME="${VIDEO_NAME:-zeroshot_training_replay}"
SEED="${SEED:-400}"
TEST_EPISODES="${TEST_EPISODES:-3}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
CKPTS=(${CKPTS:-ckpt.0 ckpt.4 final_from_resume})
if [ -z "${DATA_PATH+x}" ]; then
  DATA_PATH='data/datasets/pointnav/heldout-vangogh/v1/{split}/{split}.json.gz'
fi

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/habitat-lab}"
cd "$PROJECT_DIR"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate habitat

mkdir -p "logs/${VIDEO_NAME}" "tb/${VIDEO_NAME}" "video_dir/${VIDEO_NAME}"

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

for ckpt in "${CKPTS[@]}"; do
  ckpt_path="data/checkpoints/${EXP_NAME}/seed_${SEED}/${ckpt}.pth"
  if [ ! -f "$ckpt_path" ]; then
    echo "missing checkpoint: ${ckpt_path}" >&2
    exit 1
  fi

  echo "record checkpoint=${ckpt_path}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m habitat_baselines.run \
    --config-name=pointnav/ppo_pointnav_example.yaml \
    habitat_baselines.evaluate=True \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.eval.use_ckpt_config=False \
    habitat_baselines.eval_ckpt_path_dir="$ckpt_path" \
    habitat_baselines.test_episode_count="$TEST_EPISODES" \
    habitat_baselines.num_environments=1 \
    habitat_baselines.tensorboard_dir="tb/${VIDEO_NAME}/${ckpt}" \
    habitat_baselines.video_dir="video_dir/${VIDEO_NAME}/${ckpt}" \
    "habitat_baselines.eval.video_option=[disk]" \
    habitat_baselines.eval.split="$EVAL_SPLIT" \
    habitat.dataset.split="$EVAL_SPLIT" \
    "habitat.dataset.data_path='${DATA_PATH}'" \
    habitat.seed="$SEED" \
    >"logs/${VIDEO_NAME}/${ckpt}.log" 2>&1
done

find "video_dir/${VIDEO_NAME}" -type f | sort
