#!/bin/bash
#SBATCH --job-name pdc-ergoC-always
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoC-always-%j.out

# F3 upper-bound arm: unbudgeted always-reset (constant_remind), no
# remind-budget on purpose -- "allowed to be better because it's more
# expensive" (signal_resolution_plan.md section 4.3's arm table). Report
# sum(inserted_tokens)/n_trajectories alongside success rate, not just the
# reset count.

set -euo pipefail
module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
nvidia-smi -L || true
python -m pip install -e . -q

ITEM_IDS=$(cat conf/experiment/ergo_phaseC_item_ids.txt)
python scripts/run_ergo_math_screening.py \
  --agent-model Qwen/Qwen3-4B --device cuda \
  --output-dir outputs/ergo_math_phaseC_always_reset \
  --controller constant_remind \
  --item-ids $ITEM_IDS \
  --seeds 0 1
