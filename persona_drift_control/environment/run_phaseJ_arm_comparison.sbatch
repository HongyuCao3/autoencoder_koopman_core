#!/bin/bash
#SBATCH --job-name pdc-phaseJ-compare
#SBATCH --cpus-per-task 2
#SBATCH --mem 4G
#SBATCH --time 00:15:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseJ-compare-%j.out

# Cross-arm comparison for Phase J, re-run after the seed expansion to 5 seeds
# (docs/experiments/budget_constrained_defense_plan.md section 10.7 ->
# docs/next_step_diagnosis.md section 4 step 3). Same two invocations that
# produced the 2-seed numbers in section 10; those artifacts were copied to
# *_2seed.json first, so both are on disk side by side.
#
# CPU-only; goes through sbatch only because the code-writing sandbox has no
# python (RUNNING_ON_PALMETTO.md).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

python -m pip install -e . -q

echo "=== adaptive arm vs the best FIXED allocation ==="
python -u scripts/analyze_budget_arm_comparison.py

echo
echo "=== adaptive arm vs threshold (the other feedback controller) ==="
python -u scripts/analyze_budget_arm_comparison.py \
  --fixed-arm-prefix threshold \
  --out-path outputs/koopman_case_study/budget_arm_comparison_vs_threshold.json
