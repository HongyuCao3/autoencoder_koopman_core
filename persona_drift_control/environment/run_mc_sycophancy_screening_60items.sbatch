#!/bin/bash
#SBATCH --job-name pdc-mc-sycophancy-60
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 03:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/mc-sycophancy-60-%j.out

# Sample-expansion run of the MMLU-based sycophancy screening, following up
# on job 15567606's 20-item validation run (run_mc_sycophancy_screening.sbatch),
# which confirmed the pipeline works end-to-end (92.5% regex extraction hit
# rate, 0.5% parse failures) and found a strong clean new-Q3 inertia signal
# (r=0.42 gated) but left new-Q1/discrete-flip judgments underpowered at
# n=20. docs/experiments/mc_sycophancy_screening_pilot.md's next-step 1.
#
# 60 items x 2 seeds x 5 turns = 600 turns, 3x the validation run's 200 --
# --time scaled up accordingly (that run took 38 minutes). Same
# --item-rng-seed 0 as the validation run for consistency with this
# project's convention of reusing the same seed across scales
# (select_screening_items's stratified sampling naturally picks a mostly-
# disjoint set of items at num-items=60 vs 20 since it draws from all 57
# MMLU topics rather than 20 of them). Separate --output-dir so the
# validation run's data/report stay on disk for comparison.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_mc_sycophancy_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/mc_sycophancy_screening_60items \
  --num-items 60 --seeds 0 1 --item-rng-seed 0
