#!/bin/bash
# Re-run ONLY the upper MAPPO stage (4 seeds, concurrent) with the geodesic-
# caching speedup (geo_every=4), against the already-frozen 4 lowers; then make
# the 4-seed shadow plots for both layers.  Used after killing the slow original
# upper runs.
set -u
cd /root/autodl-tmp/habitat-lab
source /root/miniconda3/etc/profile.d/conda.sh && conda activate habitat
export PROJECT_DIR=$(pwd)

SEEDS="100 200 300 400"
UPPER_STEPS=500000
mkdir -p results/upper_mappo

echo "[upper4] ===== upper MAPPO (swap, geo_every=4), 4 seeds concurrent ===== $(date)"
pids=()
for s in $SEEDS; do
  python dhrl_habitat/upper_mappo_train.py --algorithm dhrl --seed $s \
    --lower-ckpt data/checkpoints/lower_nav/lower_nav_seed_${s}_converged.pt \
    --num-envs 4 --total-steps $UPPER_STEPS --goal-mode swap --log-every 2000 \
    > results/upper_mappo/train_dhrl_seed${s}.log 2>&1 &
  pids+=($!); echo "  upper seed $s -> PID $!"; sleep 4
done
for p in "${pids[@]}"; do wait $p; done
echo "[upper4] all upper finished $(date)"

echo "[upper4] ===== 4-seed shadow plots ====="
for m in reward success_rate; do
  python dhrl_habitat/plot_curves.py --results-dir results/lower_nav --algorithms lower_nav \
    --metric $m --raw-x --title "Lower: single-agent PPO nav (4 seeds)" \
    --out results/lower_nav/lower_${m}_4seed.png 2>&1 | grep -iE "saved|ms/step"
  python dhrl_habitat/plot_curves.py --results-dir results/upper_mappo --algorithms dhrl \
    --metric $m --raw-x --title "Upper MAPPO: commands to frozen lower (swap, 4 seeds)" \
    --out results/upper_mappo/upper_${m}_4seed.png 2>&1 | grep -iE "saved|ms/step"
done
echo "[upper4] DONE-ALL $(date)"
