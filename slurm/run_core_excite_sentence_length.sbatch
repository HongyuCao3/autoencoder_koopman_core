#!/bin/bash
#SBATCH --job-name pdc-core-excite-sentlen
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 2:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/slurm/logs/core-excite-sentlen-%j.out

# #4b PERSISTENT-EXCITATION ARM -- NOT SUBMITTED; WAITING FOR THE USER'S RULING.
# Spec: docs/experiments/command_channel_identifiability_plan.md section 3.
#
# WHY. On every core task the command is the tracking error r - y_t, an exact
# affine function of the state, so B is undefined (headroom section 2). 4a found
# no other source: the control-line logs are not in KOMPAS-code-release.zip.
# This arm breaks the identity: from turn 2 on, the word count the feedback asks
# for is a uniform draw from [5, 100], recorded per row as command_raw_count /
# command_norm. Turn-1 prompts are the landed ones verbatim. Decoding is the
# landed greedy config, so the three seeds differ ONLY in the command schedule.
#
# --disable-thinking IS REQUIRED. The landed corpus has no reasoning blocks
# (0/2520 rows); the 09-17 re-collection (16035914) ran Qwen3-4B with thinking on,
# hit the 256-token cap on 90.6% of sentence_length_t10 rows, and counted words
# inside <think>. That arm is not a valid collection; this one must not repeat it.
#
# SCALE 3 seeds x 252 trajectories x 10 turns = 7,560 generations: the landed
# protocol (T=10, 168/42/42 train/val/test). The spec's "42 x 6 x 3" did not
# match it; user ruling 2026-10-25: run the actual protocol.
#
# RUNTIME 0.25-0.75 h. ASSUMPTIONS: 16035914 ran this task on this node class in
# ~8 min including one engine start (~2 min), WITH thinking on and 90.6% of rows
# running to the 256-token cap; with thinking off, a one-sentence reply is
# ~10-120 tokens, so generation is at most that 6 min per seed. 3 seeds x
# (2 min engine + <= 6 min generation) = <= 24 min; band +/- 2x. --time 2:00:00.
#
# ENV: /scratch/hcao2/envs/sequor_vllm was rebuilt 2026-11-05 after the scratch
# purge (vllm 0.28.0, torch 2.13.0+cu130, transformers 5.16.1); the broken copy
# is kept at sequor_vllm.broken-20261105. Dry-run passed again in the new env.
# PYTHONNOUSERSITE=1 IS REQUIRED: ~/.local/lib/python3.11 holds an old cu12
# stack whose nccl shadows torch's (ImportError: undefined symbol ncclCommResume).
#
# QUOTA: method (identification data). Last 10 project jobs: instrument 1.
# ARTIFACTS: persona_drift_control/outputs/core_excite_sentence_length_<seed>/,
# none exists; the runner refuses an existing trajectories.jsonl.
# NO pip install. Uses vLLM: do not run beside another vLLM job.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core

nvidia-smi -L || true

for SEED in 0 1 2; do
  echo "=== sentence_length_t10 random command seed $SEED ==="
  python scripts/run_core_collection_arm.py \
    --task sentence_length_t10 \
    --agent-model Qwen/Qwen3-4B \
    --command-policy random \
    --command-seed "$SEED" \
    --disable-thinking \
    --out-dir persona_drift_control/outputs/core_excite_sentence_length_$SEED
done
