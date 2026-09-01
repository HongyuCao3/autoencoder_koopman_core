#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseF-mpc
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseF-mpc-%j.out

# Phase F, koopman_mpc arm -- same fitted model as Phase E
# (outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json,
# "richer_abs_sign", nu=1/mu=2, horizon=2, no repeat penalty), deployed
# unmodified against benign MT-Bench sessions instead of held-out attacks.
# See run_koopman_defense_phaseF_zero_control.sbatch.

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
  --output-dir outputs/koopman_defense_phaseF_koopman_mpc \
  --seeds 0 1 \
  --controller koopman_mpc \
  --koopman-model-path outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json \
  --koopman-model-key richer_abs_sign --koopman-nu 1 --koopman-mu 2 \
  --koopman-horizon 2 --koopman-repeat-penalty 0.0
