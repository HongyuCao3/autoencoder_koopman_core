#!/bin/bash
#SBATCH --job-name pdc-ergoC-randsched
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoC-randsched-%j.out

# F3 equal-cost random-schedule arm (spend_prob=1.0 -> exactly 1 reset per
# trajectory, uniformly random turn, matches the mpc arm's budget of 1).
# random_schedule is handled by a local branch in
# run_ergo_math_screening.py (not controller_cli.make_controller_factory):
# each item's own num_shards sets its own turn range 1..num_shards.

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
  --output-dir outputs/ergo_math_phaseC_randsched_p100 \
  --controller random_schedule --random-schedule-spend-prob 1.0 --remind-budget 1 \
  --item-ids $ITEM_IDS \
  --seeds 0 1
