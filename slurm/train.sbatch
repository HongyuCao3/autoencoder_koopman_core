#!/bin/bash
# Single Koopman training run on Palmetto 2 (Slurm).
#
#   sbatch slurm/train.sbatch dataset=sentence_length_t10 state=memory state.lag=3 trainer.epochs=400
#
# Everything after the script name is passed verbatim to scripts/train.py as
# Hydra overrides. Resource flags below are defaults; override on the command
# line *before* the script name, e.g.
#   sbatch --time 02:00:00 --gpus a100:1 slurm/train.sbatch trainer.device=cuda ...
#
# Preemption/timeout are safe: train.py installs a SIGTERM handler that writes
# a complete checkpoint and exits 130; Palmetto requeues batch jobs by default,
# and re-running the same command resumes from the latest complete checkpoint.
# (So a FAIL e-mail with exit code 130 after a preemption is expected.)

#SBATCH --job-name koopman-train
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16gb
#SBATCH --time 01:00:00
#SBATCH --signal=B:TERM@120          # SIGTERM 120s before walltime so the checkpoint can be written
#SBATCH --output slurm/logs/%x-%j.out
#SBATCH --mail-type END,FAIL,REQUEUE,TIME_LIMIT_90
# Uncomment for GPU runs (joint mode / large latent_dim sweeps); add trainer.device=cuda:
##SBATCH --gpus a100:1

set -euo pipefail

# ---- environment (edit to match `slurm/README` / your setup) ----------------
KOOPMAN_ENV="${KOOPMAN_ENV:-/scratch/$USER/envs/autoencoder-koopman-core}"
module load anaconda3
source activate "$KOOPMAN_ENV"

# hydra.job.chdir=false: relative paths (datasets/, results/) resolve against
# the cwd, so always run from the repository root.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "[slurm] job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi -L 2>/dev/null || echo "[slurm] no GPU allocated"

# `srun` registers a job step (per-step accounting in sacct) and forwards
# SIGTERM to python. `-u` keeps the log unbuffered for `tail -f`.
srun --ntasks 1 python -u scripts/train.py "$@"
