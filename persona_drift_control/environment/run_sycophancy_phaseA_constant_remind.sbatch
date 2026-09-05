#!/bin/bash
#SBATCH --job-name pdc-sycophancy-phaseA-remind
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sycophancy-phaseA-remind-%j.out

# Phase A of docs/feasibility/SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md section 5
# step 2: executor-authority check for consistency_reminder.py's channel-A
# reminder. --controller constant_remind (reminder inserted every turn,
# including turn 1 -- same convention as the adversarial-defense line's own
# Phase A, run_koopman_defense_phaseA_constant_remind.sbatch, which shares
# this trajectory runner). Compare against the sibling
# *_zero_control.sbatch run's new-Q1/new-Q3/discrete_flip_events on the SAME
# 11-item panel and seeds -- if constant_remind doesn't measurably raise
# y_consistency over zero_control here, there is no point building a Koopman
# closed loop on top of this actuator.
#
# Same 11 ground-truth-clean items, same independent judge, same --time
# budget as the zero_control sibling -- see that script's header for why.

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
  --output-dir outputs/sycophancy_phaseA_constant_remind \
  --seeds 0 1 \
  --item-ids sycon_fp_0010 sycon_fp_0024 sycon_fp_0055 sycon_fp_0066 sycon_fp_0098 \
             sycon_fp_0103 sycon_fp_0107 sycon_fp_0122 sycon_fp_0124 sycon_fp_0149 sycon_fp_0194 \
  --controller constant_remind
