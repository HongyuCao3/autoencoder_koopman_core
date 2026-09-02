#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseH-interaction-benign
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseH-interaction-benign-%j.out

# Phase H benign helpfulness-cost check for the interaction-augmented
# koopman_mpc controller (same protocol as Phase F/G's other controllers,
# same 16 fixed (benign_id, seed) MT-Bench sessions) -- the question this
# answers: does the interaction fix's documented open concern (repeat_penalty
# prunes reminders on the WORSE, not the better, states -- see
# docs/experiments/koopman_case_study_design.md) translate into any
# measurable helpfulness cost on completely benign traffic, where the
# controller's decisions are driven by the same y_safety judge signal but
# nothing is actually adversarial.

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
  --output-dir outputs/koopman_defense_phaseH_koopman_mpc_interaction_benign \
  --seeds 0 1 \
  --controller koopman_mpc_interaction \
  --koopman-nu 1 --koopman-mu 2 --koopman-horizon 2 --koopman-repeat-penalty 0.2
