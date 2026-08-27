#!/bin/bash
# Hyper-parameter / dataset sweep as a Slurm job array.
#
#   sbatch --array=1-$(wc -l < slurm/sweep_example.txt)%8 slurm/sweep_array.sbatch slurm/sweep_example.txt
#
# Each non-empty, non-comment line of the sweep file is one set of Hydra
# overrides for scripts/train.py; array index N runs line N (1-based).
# `%8` caps concurrent tasks to keep fair-share consumption gentle.
# Re-run only failed tasks with e.g. --array=7,23.
#
# run_name is derived from dataset/state/mode/latent_dim/seed, so rows that
# differ only in learning_rate, weight_decay, lambda_* or hidden_dim MUST set
# an explicit trainer.run_name=..., otherwise train.py refuses to reuse the
# checkpoint directory of the earlier row.

#SBATCH --job-name koopman-sweep
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16gb
#SBATCH --time 01:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output slurm/logs/%x-%A_%a.out
#SBATCH --mail-type END,FAIL,REQUEUE,TIME_LIMIT_90
##SBATCH --gpus a100:1

set -euo pipefail

SWEEP_FILE="${1:?usage: sbatch --array=1-N slurm/sweep_array.sbatch <sweep_file>}"
KOOPMAN_ENV="${KOOPMAN_ENV:-/scratch/$USER/envs/autoencoder-koopman-core}"
module load anaconda3
source activate "$KOOPMAN_ENV"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# strip comments/blank lines, then pick line SLURM_ARRAY_TASK_ID
LINE=$(grep -vE '^\s*(#|$)' "$SWEEP_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then
  echo "[slurm] no sweep line for index $SLURM_ARRAY_TASK_ID in $SWEEP_FILE" >&2
  exit 1
fi
echo "[slurm] array=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$(hostname)"
echo "[slurm] overrides: $LINE"

# shellcheck disable=SC2086  # word-splitting of $LINE into separate overrides is intended
srun --ntasks 1 python -u scripts/train.py $LINE
