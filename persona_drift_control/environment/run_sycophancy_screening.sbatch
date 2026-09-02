#!/bin/bash
#SBATCH --job-name pdc-sycophancy-screening
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sycophancy-screening-%j.out

# Mirrors run_adversarial_screening.sbatch. Independent of every other
# screening job (own output dir). 20 items x 2 seeds x 5 turns = ~200 turns,
# 1 agent generation + 1 short judge generation per turn -- same cost shape
# as adversarial_screening.py's step-1 pilot, so the same --time budget
# applies. --time is a generous upper bound; the resumable trajectories.jsonl
# (sycophancy_screening.py) means a --time kill just needs `sbatch` on this
# same script again.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_sycophancy_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/sycophancy_screening
