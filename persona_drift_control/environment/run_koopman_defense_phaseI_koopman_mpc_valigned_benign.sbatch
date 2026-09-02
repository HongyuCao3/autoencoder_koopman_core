#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseI-valigned-benign
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseI-valigned-benign-%j.out

# Phase I benign helpfulness-cost check for the v-aligned interaction
# koopman_mpc controller (same protocol as Phase F/G/H's other controllers,
# same 16 fixed (benign_id, seed) MT-Bench sessions) -- see
# run_koopman_defense_phaseI_koopman_mpc_valigned.sbatch for the full
# context on this fix.

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
  --output-dir outputs/koopman_defense_phaseI_koopman_mpc_valigned_benign \
  --seeds 0 1 \
  --controller koopman_mpc_interaction \
  --koopman-nu 1 --koopman-mu 2 --koopman-horizon 2 --koopman-repeat-penalty 0.0 \
  --koopman-contemporaneous-v \
  --koopman-interaction-model-path outputs/koopman_case_study/interaction_model_report_valigned.json
