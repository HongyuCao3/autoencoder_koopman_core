#!/bin/bash
#SBATCH --job-name pdc-tsar-cefr-gpu2-anchor
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 2:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/tsar-cefr-gpu2-anchor-%j.out

# tsar_cefr GPU-2 ANCHOR PASS (method). NOT SUBMITTED WITHOUT A USER RULING.
#
# WHAT IT IS FOR, AND WHY IT IS A SEPARATE JOB. Table 2's reporting pass samples
# at T=0.7 so that the three seeds carry real variance -- these arms are
# deterministic policies, so greedy decoding would write three byte-identical
# copies and a standard deviation of exactly 0. That ruling has a price: GPU-1
# and the GPU-2a tree are both GREEDY, so a sampled arm's RMSE cannot be put
# beside D-2.5's bound.
#
# THE BOUND IS THE NUMBER THAT MATTERS. D-2.5 measured the leave-one-source-out
# supremum over ALL bin-measurable closed-loop policies at RMSE 0.7752, against
# the hand-written greedy_reactive at 0.7882 and the best fitted open-loop
# schedule at 0.9320. An Ours that beats the open-loop arms has cleared nothing
# greedy_reactive had not already cleared unfitted. So one pass is run in the
# tree's own regime and compared against 0.7752:
#
#   greedy decoding        matches GPU-1 and the tree
#   one seed               a deterministic policy has nothing to average
#   EXACTLY 4 steps        the tree's depth, so the horizon matches too
#   --stop-on-arrival never  the tree generated `copy`; it did not stop
#
# THREE ARMS, NOT FIVE. The bound is a statement about the four RELATIVE control
# actions, so only the arms inside that class can be held against it:
# fixed_ladder, greedy_reactive, koopman_mpc. `one_shot` and `dp_path` both name
# a CEFR level and live outside the class -- D-2.5's open-loop supremum does NOT
# dominate them, and pretending otherwise would be the overclaim this file
# exists to prevent.
#
# THIS PASS DOES NOT ENTER TABLE 2. It answers one question -- did Ours reach
# the bound -- and it is labelled as such in the results archive. Reporting it
# as a headline would be picking the friendlier of two decodings after the fact.
#
# SCALE: 199 items x 3 arms x 1 seed x exactly 4 steps = 2,388 generations and
# the same number of readout texts, plus 199 source readouts.
#
# RUNTIME ESTIMATE 0.2-0.3 h (--time 2:00:00 is deliberate slack on a short job).
# Rates measured on GPU-2a (15897788), as in the reporting job:
#   - GENERATION 0.0128 s/gen -> 2,388 x 0.0128 = 0.5 min; band 0.5-1.2 min.
#   - READOUT 0.111-0.20 s/text -> 2,587 x (0.111-0.20) = 5-9 min. Four batches
#     of ~600 here, smaller than GPU-2a's single batch of 67,660.
#   - MPC PLANNING, CPU: 340 operator rollouts x 199 items x 4 steps = 271k
#     4x4 updates -> under a minute. NOT MEASURED.
#   - MODEL LOAD + ENGINE INIT ~3 min.
#
# QUOTA (item 5): last ten jobs are instrument 3 / method 7. This job is METHOD.
#
# ARTIFACTS: outputs/tsar_cefr_s3_anchor/ is new and distinct from the reporting
# job's outputs/tsar_cefr_s3_arm/. Nothing is overwritten; operator_fit.json is
# read only.
#
# NO pip install here (item 6). If both GPU-2 jobs are queued together they still
# install nothing, so the 2026-09-08 editable-install race cannot recur.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

# PER-JOB COMPILE CACHES. vLLM's default cache root is ~/.cache/vllm on the shared
# NFS home, and its key is the model+config hash -- so two concurrent jobs on the
# same model write the same tree from two nodes and one of them gets ENOSTALE.
# That is exactly what killed 15943452 at 3m33s while 15943453 survived
# (node2087 vs node2093, cache writes at 21:26:17 and 21:26:25). Same failure class
# as the 2026-09-08 concurrent editable install (.claude/experiments.md item 6):
# two parallel jobs mutating one shared path. Scratch, scoped by job id, is private.
export VLLM_CACHE_ROOT=/scratch/hcao2/vllm_cache/job_${SLURM_JOB_ID:-local}
export TORCHINDUCTOR_CACHE_DIR=/scratch/hcao2/inductor_cache/job_${SLURM_JOB_ID:-local}
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_tsar_cefr_closed_loop_arms.py \
  --agent-model Qwen/Qwen3-4B \
  --split test \
  --arms fixed_ladder greedy_reactive koopman_mpc \
  --seeds 0 \
  --max-steps 4 \
  --decoding greedy \
  --stop-on-arrival never \
  --beta 1.8333 \
  --lambda-cost 0.1156 \
  --operator outputs/tsar_cefr_phase3/operator_fit.json \
  --readout-device 0 \
  --readout-batch-size 32 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --mode canonical \
  --out-dir outputs/tsar_cefr_s3_anchor
