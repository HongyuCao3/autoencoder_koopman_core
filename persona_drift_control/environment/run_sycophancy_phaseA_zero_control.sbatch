#!/bin/bash
#SBATCH --job-name pdc-sycophancy-phaseA-zero
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sycophancy-phaseA-zero-%j.out

# Phase A of docs/feasibility/SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md section 5
# step 2: executor-authority check for consistency_reminder.py's channel-A
# reminder. This is the --controller zero_control half of the pair (see the
# sibling *_constant_remind.sbatch); together they answer "does the reminder
# move y_consistency at all", the precondition the feasibility doc calls the
# real first gate before any closed loop is worth building.
#
# Runs on the 11 ground-truth-clean items from
# persona_drift_control/scripts/audit_ground_truth_quality.py (docs/
# experiments/sycophancy_screening_pilot.md's ground-truth audit), NOT the
# original 20-item panel -- a null result here must not be explainable by
# the 8/20 items with bad `correction` ground truth. --judge-model is the
# independent checkpoint (default since that audit's self-judging-bias
# finding), so this is a fresh run, not comparable item-for-item to the
# earlier 20-item screening reports.
#
# --time mirrors run_sycophancy_screening_independent_judge.sbatch (same per-
# turn cost shape, smaller item count so if anything this finishes faster).
# Resumable (sycophancy_screening.py) -- a --time kill just needs `sbatch`
# on this same script again.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_sycophancy_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/sycophancy_phaseA_zero_control \
  --seeds 0 1 \
  --item-ids sycon_fp_0010 sycon_fp_0024 sycon_fp_0055 sycon_fp_0066 sycon_fp_0098 \
             sycon_fp_0103 sycon_fp_0107 sycon_fp_0122 sycon_fp_0124 sycon_fp_0149 sycon_fp_0194 \
  --controller zero_control
