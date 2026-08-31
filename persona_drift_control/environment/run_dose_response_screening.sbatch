#!/bin/bash
#SBATCH --job-name pdc-dose-response
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/dose-response-%j.out

# Step 2 (ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7): 20 queries x 5
# alpha levels = 100 single-turn generations + 100 short judge generations.
# Requires run_calibrate_safety_direction.sbatch to have completed first
# (outputs/safety_direction/safety_direction.npy must exist) -- --layer here
# must match the layer that job calibrated at. --time is a generous upper
# bound (100 single-turn generations, no multi-turn replay, so should be
# cheaper than run_adversarial_screening's 200 multi-turn ones) -- the
# resumable dose_response_rows.jsonl (dose_response_screening.py) means a
# --time kill just needs `sbatch` on this same script again.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_dose_response_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --layer 18 \
  --direction-path outputs/safety_direction/safety_direction.npy \
  --output-dir outputs/dose_response
