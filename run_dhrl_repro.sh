#!/usr/bin/env bash
# D-HRL paper reproduction pipeline (advisor-aligned, unified Habitat).
# Run on AutoDL from the project root (PROJECT_DIR), inside the habitat conda env.
#
#   bash run_dhrl_repro.sh preflight   # ALWAYS run first (validates new configs)
#   bash run_dhrl_repro.sh stage1      # train lower operator (velocity mode)
#   bash run_dhrl_repro.sh validate    # manual-command gate (人工指令验证)
#   bash run_dhrl_repro.sh stage2      # 3 algorithms x 4 seeds MARL
#   bash run_dhrl_repro.sh plots       # shadow curves + inference-time table
#   bash run_dhrl_repro.sh all
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
EXP_NAME="${EXP_NAME:-dhrl_repro}"
CONFIG="${CONFIG:-benchmark/multi_agent/hssd_spot_spot.yaml}"
LOWER_MODE="${LOWER_MODE:-velocity}"
LOWER_STEPS="${LOWER_STEPS:-500000}"
UPPER_STEPS="${UPPER_STEPS:-2000000}"
SEEDS="${SEEDS:-100 200 300 400}"
ALGOS="${ALGOS:-dhrl no_memory no_hierarchy}"
N_AGENTS="${N_AGENTS:-2}"
GOAL_MODE="${GOAL_MODE:-swap}"
LOWER_SEED="${LOWER_SEED:-100}"
LOWER_CKPT="${LOWER_CKPT:-data/checkpoints/${EXP_NAME}/lower_${LOWER_MODE}_seed_${LOWER_SEED}.pt}"

cd "$PROJECT_DIR"
export PROJECT_DIR

stage_preflight() {
  python dhrl_habitat/preflight.py --config-name "$CONFIG" --n-agents "$N_AGENTS"
}

stage_1() {
  python dhrl_habitat/lower_train.py \
    --config-name "$CONFIG" --exp-name "$EXP_NAME" \
    --mode "$LOWER_MODE" --seed "$LOWER_SEED" --total-steps "$LOWER_STEPS"
}

stage_validate() {
  python dhrl_habitat/manual_validate.py \
    --config-name "$CONFIG" --lower-ckpt "$LOWER_CKPT" \
    --n-agents "$N_AGENTS" --goal-mode "$GOAL_MODE" --episodes 50
}

stage_2() {
  for algo in $ALGOS; do
    for seed in $SEEDS; do
      echo "=== stage2 algorithm=$algo seed=$seed ==="
      args=(--config-name "$CONFIG" --exp-name "$EXP_NAME"
            --algorithm "$algo" --seed "$seed"
            --n-agents "$N_AGENTS" --goal-mode "$GOAL_MODE"
            --total-steps "$UPPER_STEPS")
      if [ "$algo" != "no_hierarchy" ]; then
        args+=(--lower-ckpt "$LOWER_CKPT")
      fi
      python dhrl_habitat/upper_train.py "${args[@]}" \
        2>&1 | tee "results/${EXP_NAME}/${algo}_seed_${seed}.log"
    done
  done
}

stage_plots() {
  python dhrl_habitat/plot_curves.py --results-dir "results/${EXP_NAME}" \
    --algorithms $ALGOS --metric reward
  python dhrl_habitat/plot_curves.py --results-dir "results/${EXP_NAME}" \
    --algorithms $ALGOS --metric success_rate
}

mkdir -p "results/${EXP_NAME}"
case "${1:-all}" in
  preflight) stage_preflight ;;
  stage1)    stage_1 ;;
  validate)  stage_validate ;;
  stage2)    stage_2 ;;
  plots)     stage_plots ;;
  all)       stage_preflight; stage_1; stage_validate; stage_2; stage_plots ;;
  *) echo "usage: $0 {preflight|stage1|validate|stage2|plots|all}"; exit 1 ;;
esac
