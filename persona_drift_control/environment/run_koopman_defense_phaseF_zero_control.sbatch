#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseF-zero
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseF-zero-%j.out

# Phase F of docs/experiments/koopman_defense_pilot.md: benign
# helpfulness-cost check. This is the zero_control (no defense) arm -- the
# baseline every other Phase F arm's y_help is compared against
# (analysis_helpfulness.compare_arms_to_zero_control). Runs all 8 MT-Bench
# categories x 2 seeds = 16 sessions x 6 turns, with two judge calls per turn
# (safety + helpfulness), roughly double Phase E's per-turn judge cost -- see
# the longer time budget vs Phase E's sbatch files.

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
  --output-dir outputs/koopman_defense_phaseF_zero_control \
  --seeds 0 1 \
  --controller zero_control
