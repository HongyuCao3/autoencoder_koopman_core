#!/bin/bash
#SBATCH --job-name pdc-gen-user-scripts
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/gen-user-scripts-%j.out

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/generate_user_scripts.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --seeds 0 1 2 3
