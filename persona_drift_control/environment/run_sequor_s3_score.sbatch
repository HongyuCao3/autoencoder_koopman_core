#!/bin/bash
#SBATCH --job-name pdc-sequor-s3-score
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 03:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s3-score-%j.out

# S3's REPORTING READOUT (instrument). Submit with
#   --dependency=afterok:<s3 arm job id>
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md).
#
# WHAT IT DECIDES: where every S3 number comes from. The arm already carries
# the in-loop 4B verdicts that DROVE the controller; those are a selection
# signal and are never reported (.claude/global.md -> reporting standard). This
# job re-judges the same rows with the independent Qwen3-14B that S0
# calibrated (agreement 0.8944, kappa 0.7886 on 2000 gold rows), and its
# `followed` / `y_graded` fields are the only source of a reported number.
#
# A SEPARATE JOB on purpose: re-judging is cheap and re-generating is not.
# Job 15763577 rescued 15761810's 52 GPU-minutes for exactly this reason.
#
# RUNTIME ESTIMATE 0.9-1.4 h (--time 03:00:00). ASSUMPTIONS: 9,000 rows x 3
# constraints = 27,000 judge calls at the 14B's measured 0.11 s/call
# (15772957, 28,800 calls) -> ~50 min, + ~4 min load. cap 512 is sufficient
# for the 14B (12/2000 truncated at 15739195) and is NOT raised here.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/score_sequor_trajectories.py \
  --arm-dir outputs/sequor_s3_arm \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --max-model-len 8192 \
  --out-path outputs/sequor_s3_arm/readout_independent_qwen3_14b.json
