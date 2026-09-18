#!/bin/bash
#SBATCH --job-name pdc-core-arms-qwen3_4b
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 4:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/core-arms-qwen3_4b-%j.out

# SECOND BACKBONE, the five admitted core tasks, run as ONE backbone's half of a
# PAIR. Model: Qwen/Qwen3-4B
# Plan: docs/experiments/second_backbone_plan_2026-09-17.md
#
# WHY A PAIR AND NOT A COMPARISON AGAINST THE PUBLISHED QWEN COLUMN. The core
# corpus has no generator in this repository; this arm is collected by
# scripts/run_core_collection_arm.py, which REPLAYS a protocol recovered from the
# landed rows (turn structure, scorer, feedback rule, normalisation -- all four
# legs verified row-exact before the runner existed). Two of the five tasks
# score at 0.9927-1.0000 rather than exactly 1.0000, and a reconstructed harness
# is not the collaborator's harness even where it reproduces it. So the
# comparison lives INSIDE this group: the same runner, twice, once per backbone.
# The published Qwen column is not the control. Precedent: G1 rebuilt all five
# seeds instead of continuing, to buy one harness fingerprint.
#
# DECODING DOES NOT DRIFT HERE, unlike the constraint line. The runner takes the
# decoding from the landed decoding_config, and core was collected GREEDY
# (do_sample false, temperature 0.0). Both halves of the pair therefore run the
# same decoding, and the only thing that moves is the weights.
#
# ROW 3 IS STRUCTURALLY ZERO ON CORE AND STAYS ZERO. The control is the tracking
# error r - y_t, an exact affine function of the state (R^2 = 1.0 on all eight
# tasks). This pair supports Table 1 rows 1/2/4/5/6 only; it cannot speak about
# the actuator. Nothing in this job changes that and no rerun will.
#
# even_odd_t5 CARRIES A KNOWN HARD-FAIL (user ruling 2026-09-17). All 96 landed
# feedback rows say "Your answer was correct", so the wrong-answer branch has no
# witness and cannot be recovered from the data. The builder RAISES on it rather
# than guessing. If this backbone ever answers wrong, that task dies here and is
# reported as "not replayable in this group" -- the other four are unaffected
# because each task is a separate invocation.
#
# SCALE 7,740 generations: sentence_length_t10 2,520 + character_length_t5 1,800
# + average_word_length_t5 420 + even_odd_t5 120 + vector_count_stage1_t10 2,880.
# All short (landed caps 16-256 tokens), batched per turn across 24-360
# trajectories.
#
# RUNTIME 0.5-1.5 h. ASSUMPTIONS: the tsar_cefr gemma arm (16018827) did 3,640
# generations end-to-end in 10m48 on this node under the same vLLM version, so
# ~7,740 short ones is ~20-25 min of generation; plus FIVE engine starts (one
# per task invocation, ~3 min each on the tsar_cefr evidence) = ~15 min. The
# band is that sum +/- 50%. --time 4:00:00.
#
# QUOTA: method.
# ARTIFACTS: outputs/core_qwen3_4b_<task>/ are all new; datasets/ is read-only
# to this runner and the published core columns are untouched.
# NO pip install (item 6). USES vLLM, so it must not run beside another vLLM job
# -- 15943452 died in 3m33s on a shared compile cache. This job is submitted
# with an explicit --dependency chain for exactly that reason, not with a hope.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
# 16021887/16021888 both died here at engine init: vLLM's default sampler path
# JIT-builds flashinfer, and the compute nodes have no nvcc
# ("Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist").
# The three vLLM jobs that DID run today (16018827, 16021868) carry both of these.
# Nothing else about the arm changes.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core

nvidia-smi -L || true

for TASK in sentence_length_t10 character_length_t5 average_word_length_t5 even_odd_t5 vector_count_stage1_t10; do
  echo "=== $TASK (qwen3_4b) ==="
  python scripts/run_core_collection_arm.py \
    --task "$TASK" \
    --agent-model Qwen/Qwen3-4B \
    --out-dir persona_drift_control/outputs/core_qwen3_4b_$TASK
done
