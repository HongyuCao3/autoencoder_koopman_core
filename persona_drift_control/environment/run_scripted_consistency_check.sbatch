#!/bin/bash
#SBATCH --job-name pdc-consistency-check
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/consistency-check-%j.out

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

# docs/SCRIPTED_USER_TURNS_FEASIBILITY.md section 5 step 1: same 3 prompts x
# 2 seeds x 2 conditions (12 trajectories), run once with a live user-sim and
# once with the scripted library, same prompt-rng-seed so both pick the same
# prompts/topics -- compare y_probe trajectories afterwards.

echo "=== live ==="
python scripts/run_signal_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --num-prompts 3 --seeds 0 1 --prompt-rng-seed 0 \
  --user-mode live \
  --output-dir outputs/consistency_check_live

echo "=== scripted ==="
python scripts/run_signal_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --num-prompts 3 --seeds 0 1 --prompt-rng-seed 0 \
  --user-mode scripted \
  --output-dir outputs/consistency_check_scripted
