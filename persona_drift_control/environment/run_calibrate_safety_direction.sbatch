#!/bin/bash
#SBATCH --job-name pdc-safety-direction
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/safety-direction-%j.out

# Step 2 prerequisite (ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7):
# 40 forward passes (no generation) to compute the diff-in-means safety
# direction at layer 18 (half of Qwen3-4B's 36 hidden layers -- see
# docs/feasibility/LLM_LATENT_STATE_FEASIBILITY.md for the architecture numbers; not a
# swept choice yet, a documented starting point). Much cheaper than either
# screening job: no generation at all, just encodes 40 short prompts.
# Not resumable (single short job, not worth the bookkeeping) -- if killed,
# just resubmit.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/calibrate_safety_direction.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --layer 18 \
  --output-dir outputs/safety_direction
