#!/bin/bash
#SBATCH --job-name pdc-mc-phaseA-remind
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/mc-phaseA-remind-%j.out

# Executor-authority check (Phase A analog) on the MMLU sycophancy item pool
# -- see the sibling *_zero_control.sbatch for full context.
# --controller constant_remind (consistency_reminder.py's channel-A text
# inserted every turn, including turn 1 -- same convention as every other
# Phase A in this project). SAME 30 items/seeds/item-rng-seed as the
# zero_control sibling, so the two runs are a directly comparable pair.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_mc_sycophancy_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/mc_sycophancy_phaseA_constant_remind \
  --num-items 30 --seeds 0 1 --item-rng-seed 1 \
  --controller constant_remind
