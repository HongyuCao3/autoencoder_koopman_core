#!/bin/bash
#SBATCH --job-name pdc-sycophancy-continuous
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sycophancy-continuous-%j.out

# Continuous (label-token-distribution) readout backfill for the two
# already-collected sycophancy screening runs
# (docs/experiments/continuous_readout_plan.md, S3/S5).
#
# No agent generation at all: only reads outputs/sycophancy_screening/ and
# outputs/sycophancy_screening_independent_judge/'s trajectories.jsonl, and
# only writes each directory's own continuous_readout/ subdirectory -- the
# original files are never touched. ~400 rows total, each one a single
# forward pass replacing the judge's original 10-token greedy decode, so the
# real cost is the two model loads (Qwen/Qwen3-4B and
# Qwen/Qwen3-4B-Instruct-2507), not the scoring itself.
#
# --time 00:30:00: two 4B model loads plus ~400 single-token forward passes
# is well under this; resumable via continuous_readout.score_file either way.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/score_sycophancy_continuous.py --device cuda

# CPU-only, seconds -- run here too so the job's log already contains the
# G0-G3 gate report instead of needing a second manual step.
python scripts/analyze_continuous_readout.py
