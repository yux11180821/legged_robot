#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/habitat-lab}"
EXP_NAME="${EXP_NAME:-rlvigen_pointnav_camera_generalization_50ep}"
TRAIN_EXP="${TRAIN_EXP:-paper_ppo_pointnav_4seed_200k}"
TEST_EPISODES="${TEST_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-2}"
SEEDS=(${SEEDS:-100 200 300 400})

cd "$PROJECT_DIR"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate habitat

mkdir -p "logs/${EXP_NAME}" "tb/${EXP_NAME}" "outputs/${EXP_NAME}" "video_dir/${EXP_NAME}"

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

run_condition() {
  local cond="$1"
  shift
  local overrides=("$@")
  local pids=()
  echo "condition=${cond}"
  for i in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$i]}"
    local gpu="$((i % 2))"
    local ckpt="data/checkpoints/${TRAIN_EXP}/seed_${seed}/final_from_resume.pth"
    local log_file="logs/${EXP_NAME}/${cond}_seed_${seed}.log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python -m habitat_baselines.run \
        --config-name=pointnav/ppo_pointnav_example.yaml \
        habitat_baselines.evaluate=True \
        habitat_baselines.load_resume_state_config=False \
        habitat_baselines.eval.use_ckpt_config=False \
        habitat_baselines.eval_ckpt_path_dir="$ckpt" \
        habitat_baselines.test_episode_count="$TEST_EPISODES" \
        habitat_baselines.num_environments="$NUM_ENVS" \
        habitat_baselines.tensorboard_dir="tb/${EXP_NAME}/${cond}/seed_${seed}" \
        habitat_baselines.video_dir="video_dir/${EXP_NAME}/${cond}/seed_${seed}" \
        habitat_baselines.eval.video_option=[] \
        habitat.dataset.split=val \
        habitat.seed="$seed" \
        "${overrides[@]}"
    ) >"$log_file" 2>&1 &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_condition default

run_condition fov_60 \
  habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=60 \
  habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=60

run_condition fov_120 \
  habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=120 \
  habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=120

run_condition camera_low \
  'habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position=[0.0,0.75,0.0]' \
  'habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position=[0.0,0.75,0.0]'

run_condition camera_high \
  'habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position=[0.0,1.75,0.0]' \
  'habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position=[0.0,1.75,0.0]'

echo "finished"
