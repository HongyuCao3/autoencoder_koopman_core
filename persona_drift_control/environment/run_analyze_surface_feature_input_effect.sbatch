#!/bin/bash
#SBATCH --job-name pdc-surface-input-fx
#SBATCH --cpus-per-task 2
#SBATCH --mem 4G
#SBATCH --time 00:10:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/surface-input-fx-%j.out

# CPU-only, no --gpus: reads the trajectories_with_surface_features.jsonl
# that run_backfill_surface_features.sbatch already wrote, no new generation
# or feature extraction. Independent of run_screening.sbatch /
# run_drift_confirmation_pilot.sbatch, cannot contend for their GPU
# allocation.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

python -m pip install -e . -q
python scripts/analyze_surface_feature_input_effect.py \
  --augmented-path outputs/signal_screening/trajectories_with_surface_features.jsonl \
  --output-dir outputs/signal_screening
