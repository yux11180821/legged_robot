#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/habitat-lab}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-habitat}"

EXP_NAME="${EXP_NAME:-dhrl_social_nav_2algorithms}"
TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-1}"
CKPT_INTERVAL_FRAMES="${CKPT_INTERVAL_FRAMES:-100000}"
NUM_CHECKPOINTS="${NUM_CHECKPOINTS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SEEDS="${SEEDS:-100 200 300 400}"
GPUS="${GPUS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [ "${SKIP_ASSET_CHECK:-0}" != "1" ]; then
  python scripts/check_hab3_social_nav_assets.py --project-dir "$PROJECT_DIR"
fi

python train_ddp.py \
  --project-dir "$PROJECT_DIR" \
  --config-name benchmark/multi_agent/hssd_spot_human_social_nav.yaml \
  --exp-name "$EXP_NAME" \
  --algorithms dhrl no_memory \
  --seeds $SEEDS \
  --total-steps "$TOTAL_STEPS" \
  --num-envs "$NUM_ENVS" \
  --ckpt-interval-frames "$CKPT_INTERVAL_FRAMES" \
  --num-checkpoints "$NUM_CHECKPOINTS" \
  --log-interval "$LOG_INTERVAL" \
  --gpus "$GPUS" \
  --max-parallel "$MAX_PARALLEL"

python scripts/plot_dhrl_multi_agent_shadow.py \
  --tb-root "tb/${EXP_NAME}" \
  --out-dir "results/${EXP_NAME}" \
  --metric reward \
  --bin-size "$CKPT_INTERVAL_FRAMES"
