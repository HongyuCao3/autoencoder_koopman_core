#!/bin/bash
#SBATCH --job-name pdc-pytest
#SBATCH --cpus-per-task 2
#SBATCH --mem 4G
#SBATCH --time 00:10:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/pytest-%j.out

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

pytest -q
