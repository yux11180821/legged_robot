#!/usr/bin/env bash
set -euo pipefail

SCENARIOS=${SCENARIOS:-"cooperative_transport corridor_crossing ravine_bridging"}
VARIANTS=${VARIANTS:-"dhrl no_memory no_hierarchy"}
SEEDS=${SEEDS:-"100 200 300 400"}
TOTAL_STEPS=${TOTAL_STEPS:-200000}
NUM_ENVS=${NUM_ENVS:-64}
DEVICE=${DEVICE:-auto}

for scenario in $SCENARIOS; do
  for variant in $VARIANTS; do
    for seed in $SEEDS; do
      python -m dhrl.train \
        --config "configs/dhrl/${scenario}.json" \
        --variant "$variant" \
        --seed "$seed" \
        --total-steps "$TOTAL_STEPS" \
        --num-envs "$NUM_ENVS" \
        --device "$DEVICE"
    done
  done
done
