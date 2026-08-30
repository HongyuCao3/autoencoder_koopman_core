#!/bin/bash
#SBATCH --job-name pdc-surface-features
#SBATCH --cpus-per-task 2
#SBATCH --mem 4G
#SBATCH --time 00:15:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/surface-features-%j.out

# CPU-only, no --gpus requested: this backfills features onto text that
# outputs/signal_screening/trajectories.jsonl already contains, it does not
# generate anything new. Deliberately independent of run_screening.sbatch /
# run_drift_confirmation_pilot.sbatch so it cannot contend for the GPU
# allocation those jobs hold.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

python -m pip install -e . -q
python scripts/backfill_surface_features.py \
  --trajectories-path outputs/signal_screening/trajectories.jsonl \
  --output-dir outputs/signal_screening
