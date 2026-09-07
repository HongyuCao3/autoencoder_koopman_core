#!/bin/bash
#SBATCH --job-name pdc-ergoC-zero
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoC-zero-%j.out

# F3 (docs/experiments/signal_resolution_plan.md section four): ERGO Phase C
# lower-bound arm. 58 items (evaluation set, disjoint from the 45 Phase B
# identification items -- see the plan's section 4.4), 2 seeds.

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
  --output-dir outputs/ergo_math_phaseC_zero_control \
  --controller zero_control \
  --item-ids $ITEM_IDS \
  --seeds 0 1
