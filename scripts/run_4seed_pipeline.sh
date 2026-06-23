#!/bin/bash
# 4-seed full-pipeline reproduction (lower single-agent PPO nav -> freeze ->
# upper MAPPO) for paper-style mean+/-std SHADOW bands.  Seeds run concurrently
# (the box has 112 cores), so 4 seeds cost ~1x wall-clock per stage.
set -u
cd /root/autodl-tmp/habitat-lab
source /root/miniconda3/etc/profile.d/conda.sh && conda activate habitat
export PROJECT_DIR=$(pwd)

SEEDS="100 200 300 400"
LOWER_STEPS=300000
UPPER_STEPS=500000
mkdir -p results/lower_nav results/upper_mappo data/checkpoints/lower_nav data/checkpoints/upper_mappo

echo "[4seed] ===== STAGE 1: lower (single-agent PPO nav), 4 seeds concurrent ===== $(date)"
pids=()
for s in $SEEDS; do
  python dhrl_habitat/lower_nav_train.py --seed $s --num-envs 4 \
    --total-steps $LOWER_STEPS --exp-name lower_nav --log-every 2000 \
    > results/lower_nav/train_seed${s}.log 2>&1 &
  pids+=($!); echo "  lower seed $s -> PID $!"; sleep 4
done
for p in "${pids[@]}"; do wait $p; done
echo "[4seed] all lower seeds finished $(date)"

# freeze each seed's converged lower under a stable name
for s in $SEEDS; do
  cp data/checkpoints/lower_nav/lower_nav_seed_${s}_final.pt \
     data/checkpoints/lower_nav/lower_nav_seed_${s}_converged.pt
  echo "  frozen lower seed $s"
done

echo "[4seed] ===== STAGE 2: upper MAPPO (swap, vs frozen lower), 4 seeds concurrent ===== $(date)"
pids=()
for s in $SEEDS; do
  python dhrl_habitat/upper_mappo_train.py --algorithm dhrl --seed $s \
    --lower-ckpt data/checkpoints/lower_nav/lower_nav_seed_${s}_converged.pt \
    --num-envs 4 --total-steps $UPPER_STEPS --goal-mode swap --log-every 2000 \
    > results/upper_mappo/train_dhrl_seed${s}.log 2>&1 &
  pids+=($!); echo "  upper seed $s -> PID $!"; sleep 4
done
for p in "${pids[@]}"; do wait $p; done
echo "[4seed] all upper seeds finished $(date)"

echo "[4seed] ===== STAGE 3: 4-seed shadow plots ====="
for m in reward success_rate; do
  python dhrl_habitat/plot_curves.py --results-dir results/lower_nav --algorithms lower_nav \
    --metric $m --raw-x --title "Lower: single-agent PPO nav (4 seeds)" \
    --out results/lower_nav/lower_${m}_4seed.png 2>&1 | grep -iE "saved|ms/step"
  python dhrl_habitat/plot_curves.py --results-dir results/upper_mappo --algorithms dhrl \
    --metric $m --raw-x --title "Upper MAPPO: commands to frozen lower (swap, 4 seeds)" \
    --out results/upper_mappo/upper_${m}_4seed.png 2>&1 | grep -iE "saved|ms/step"
done
echo "[4seed] DONE-ALL $(date)"
