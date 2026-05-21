#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-rlvigen_pointnav_zeroshot_curve_200k}"
TEST_EPISODES="${TEST_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-2}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_NAME="${EVAL_NAME:-${EXP_NAME}_eval_vangogh_${EVAL_SPLIT}_${TEST_EPISODES}ep}"
if [ -z "${DATA_PATH+x}" ]; then
  DATA_PATH='data/datasets/pointnav/heldout-vangogh/v1/{split}/{split}.json.gz'
fi
SEEDS=(${SEEDS:-100 200 300 400})

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/habitat-lab}"
cd "$PROJECT_DIR"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate habitat

mkdir -p "logs/${EVAL_NAME}" "tb/${EVAL_NAME}" "outputs/${EVAL_NAME}" "video_dir/${EVAL_NAME}"

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

for seed in "${SEEDS[@]}"; do
  ckpt_dir="data/checkpoints/${EXP_NAME}/seed_${seed}"
  if [ ! -f "${ckpt_dir}/final.pth" ] && [ ! -f "${ckpt_dir}/final_from_resume.pth" ]; then
    echo "missing final checkpoint for seed ${seed}: ${ckpt_dir}/final.pth" >&2
    exit 1
  fi
done

mapfile -t ckpt_names < <(python - <<PY
from pathlib import Path
import torch

root = Path("data/checkpoints/${EXP_NAME}/seed_${SEEDS[0]}")
ckpts = sorted(
    root.glob("ckpt.*.pth"),
    key=lambda p: int(p.stem.split(".")[1]),
)
final = root / "final.pth"
if not final.exists():
    final = root / "final_from_resume.pth"
if final.exists():
    ckpts.append(final)

selected = []
seen_steps = set()
for path in ckpts:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    step = int(obj.get("extra_state", {}).get("step", -1))
    if step in seen_steps:
        continue
    seen_steps.add(step)
    selected.append(path.name)

for name in selected:
    print(name)
PY
)
echo "checkpoints=${ckpt_names[*]}"
echo "eval_split=${EVAL_SPLIT}"
echo "test_episodes=${TEST_EPISODES}"

for ckpt_name in "${ckpt_names[@]}"; do
  label="${ckpt_name%.pth}"
  echo "eval checkpoint=${ckpt_name}"
  pids=()
  for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="$((i % 2))"
    ckpt_path="data/checkpoints/${EXP_NAME}/seed_${seed}/${ckpt_name}"
    log_file="logs/${EVAL_NAME}/${label}_seed_${seed}.log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python -m habitat_baselines.run \
        --config-name=pointnav/ppo_pointnav_example.yaml \
        habitat_baselines.evaluate=True \
        habitat_baselines.load_resume_state_config=False \
        habitat_baselines.eval.use_ckpt_config=False \
        habitat_baselines.eval_ckpt_path_dir="$ckpt_path" \
        habitat_baselines.test_episode_count="$TEST_EPISODES" \
        habitat_baselines.num_environments="$NUM_ENVS" \
        habitat_baselines.tensorboard_dir="tb/${EVAL_NAME}/${label}/seed_${seed}" \
        habitat_baselines.video_dir="video_dir/${EVAL_NAME}/${label}/seed_${seed}" \
        habitat_baselines.eval.video_option=[] \
        habitat_baselines.eval.split="$EVAL_SPLIT" \
        habitat.dataset.split="$EVAL_SPLIT" \
        "habitat.dataset.data_path='${DATA_PATH}'" \
        habitat.seed="$seed"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [ "$status" -ne 0 ]; then
    echo "evaluation failed for ${ckpt_name}" >&2
    exit "$status"
  fi
done

echo "finished"
