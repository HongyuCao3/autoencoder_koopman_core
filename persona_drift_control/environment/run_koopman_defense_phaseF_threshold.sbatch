#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseF-thresh
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseF-thresh-%j.out

# Phase F, threshold arm (y_min=0.7 feedback, same classical baseline as
# Phase E) -- see run_koopman_defense_phaseF_zero_control.sbatch.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_benign_helpfulness_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseF_threshold \
  --seeds 0 1 \
  --controller threshold --threshold-y-min 0.7
