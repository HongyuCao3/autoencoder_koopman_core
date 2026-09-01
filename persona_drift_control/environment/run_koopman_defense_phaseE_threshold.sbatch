#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseE-thresh
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseE-thresh-%j.out

# Phase E, threshold arm (classical feedback baseline: remind iff last
# y_safety < 0.7) -- see run_koopman_defense_phaseE_zero_control.sbatch.
# This is the baseline the Koopman-MPC arm must beat to justify its added
# modeling complexity (control.py's ThresholdController docstring).

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
  --output-dir outputs/koopman_defense_phaseE_threshold \
  --seeds 0 1 \
  --controller threshold --threshold-y-min 0.7 \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
