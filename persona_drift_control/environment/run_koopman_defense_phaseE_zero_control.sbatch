#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseE-zero
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseE-zero-%j.out

# Phase E of docs/experiments/koopman_defense_pilot.md: closed-loop
# four-arm comparison. This is the zero_control (no defense) arm, run on
# the EXACT 8 held-out attack ids from Phase C's fit (not used in the
# Koopman identification data), same set as the other 3 Phase E arms, for a
# rigorous apples-to-apples new-Q1 comparison -- not a reuse of job
# 15399715's different 20-attack panel.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseE_zero_control \
  --seeds 0 1 \
  --controller zero_control \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
