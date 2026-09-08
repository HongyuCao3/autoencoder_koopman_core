#!/bin/bash
#SBATCH --job-name pdc-ergoAC-fixed_t2
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 06:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoAC-fixed_t2-%j.out

# E2 (docs/experiments/two_task_success_plan.md section 2 E2): append-mode
# fixed_t2 arm (reset exactly on turn 2, budget 1), same held-out 58 items
# as Phase C, --reset-mode append. Copied from
# environment/run_ergo_phaseC_fixed_t2.sbatch verbatim except job-name,
# log path, --reset-mode append, --output-dir, and --time (widened to
# 06:00:00: append-mode prompts grow monotonically, so this runs slower
# than overwrite's 01:00:00 template).

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
  --output-dir outputs/ergo_appendC_fixed_t2 \
  --controller fixed_schedule --fixed-schedule-turns 2 --remind-budget 1 \
  --reset-mode append \
  --item-ids $ITEM_IDS \
  --seeds 0 1
