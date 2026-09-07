#!/bin/bash
#SBATCH --job-name pdc-ergo-phaseB-excite
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergo-phaseB-excite-%j.out

# Phase B of the ERGO/Laban multi-turn-reliability line
# (docs/experiments/ergo_multiturn_reliability_pilot.md): open-loop
# random-excitation data collection for a first Koopman fit
# (scripts/fit_koopman_ergo_model.py), now that the minimal executor-
# authority check has passed twice (20-item pilot job 15602880/15602881,
# 60-item expansion job 15603367/15603368 -- both confirm large, stable
# authority, p=1.27e-4 and p=6.39e-9 respectively). Same role as
# run_koopman_defense_phaseB_random_excite.sbatch: --controller
# random_excite draws u_reset i.i.d. Bernoulli(p) each turn instead of the
# all-0/all-1 extremes the authority check used, which is what a Koopman
# identification fit needs (informative within-arm variation in the
# control input, not just between-arm).

# A fresh 60-item draw (--item-rng-seed 2, neither the pilot's 20 nor the
# 60-item expansion's --item-rng-seed 1 sample -- overlap is fine, the item
# pool only has 103 items total and no pairing constraint applies to a
# single-arm collection run). 60 items x 2 seeds, p=0.5 -- same scale as
# run_koopman_defense_phaseB_random_excite.sbatch's 30 attacks x 2 seeds,
# and same cost profile as the 60-item authority-check jobs (~15-30 min
# actual runtime observed there); --time is a generous upper bound.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_ergo_math_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/ergo_math_phaseB_random_excite \
  --num-items 60 --item-rng-seed 2 \
  --seeds 0 1 \
  --controller random_excite --random-excite-p 0.5
