#!/bin/bash
#SBATCH --job-name pdc-ergo-authority-reset
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergo-authority-reset-%j.out

# Minimal executor-authority check, reset arm -- see the sibling
# *_zero_control.sbatch for full context. --controller constant_remind
# (control.py's ConstantRemindController, reused here to mean "reset every
# turn" -- ergo_math_trajectory.py interprets a 1 from the controller as
# "wipe history and re-present every shard revealed so far as one
# consolidated message", not as a text-prefix reminder). SAME 20 items as
# the zero_control sibling, so the two runs are a directly comparable pair.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_ergo_math_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/ergo_math_authority_reset \
  --seeds 0 1 \
  --item-ids ergo_GSM8K_1247 ergo_GSM8K_825 ergo_GSM8K_1287 ergo_GSM8K_14 ergo_GSM8K_1240 \
    ergo_GSM8K_942 ergo_GSM8K_808 ergo_GSM8K_646 ergo_GSM8K_855 ergo_GSM8K_1260 ergo_GSM8K_1132 \
    ergo_GSM8K_107 ergo_GSM8K_417 ergo_GSM8K_1180 ergo_GSM8K_562 ergo_GSM8K_40 ergo_GSM8K_178 \
    ergo_GSM8K_1269 ergo_GSM8K_740 ergo_GSM8K_117 \
  --controller constant_remind
