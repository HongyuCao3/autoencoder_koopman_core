#!/bin/bash
#SBATCH --job-name pdc-smoke-test
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/smoke-test-%j.out

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_signal_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --num-prompts 1 --seeds 0 --num-turns 2 --probe-repeats 1 \
  --output-dir outputs/smoke_test
