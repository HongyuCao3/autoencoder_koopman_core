#!/bin/bash
#SBATCH --job-name pdc-sequor-s00-arm
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 02:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s00-arm-%j.out

# S0-0 COUNTERFACTUAL BRANCH ARM, CANONICAL (method -- user ruling 2026-09-09:
# classified 方法 by the EK-A precedent, since the pairs are identification
# data, not another measuring device).
# docs/experiments/constraint_signal_screening.md sections 2, 8 and 10.
#
# N=12 dialogues x T=20 turns. Every turn from the second on is generated twice
# from a byte-identical prefix -- once continuing under u=0, once with the
# constraint block restated -- so the one-step gain of a reminder is measured
# per (item, turn). 240 base + 228 counterfactual = 468 generations, 228 pairs
# (NOT 240: turn 1 already carries the block, so u=1 there is not an action --
# screening section 10 item 1).
#
# WHAT IT DECIDES: whether this line lives. K1 (readout range) failing closes
# it with no second judge attempt -- that is how `defense` died. K2 (state
# beyond the turn index) failing closes it too -- that is how ERGO died, at 97%
# attenuation. Only K1+K2+K3 together open S1. The gates are computed by
# scripts/analyze_sequor_s0_0_gates.py from the INDEPENDENT judge's readout,
# after the two scoring jobs.
#
# The 12 items are the first 12 of the 16 dialogues whose k=3 constraints are
# ALL inside the S0 gold set (screening 9.2), so a failed gate cannot also be
# blamed on the judge extrapolating to constraints it was never calibrated on.
#
# RUNTIME ESTIMATE 12-30 min, from the smoke (job 15742083, MEASURED):
#   - 78 generations took 4.1 min at 3.14 s/generation, but at a batch width of
#     only 2. This arm's batches are 12 wide, so per-generation cost falls;
#     the pessimistic bound is the smoke's rate unchanged, 468 x 3.14 = 24.5 min.
#   - responses ran to a median of 511 tokens and a max of 866, with 0/78
#     hitting the 1024 cap, so --max-new-tokens 1024 stands and the readout is
#     not measuring truncation (screening section 10 item 4; note the cap of
#     512 that the judge smoke had blessed would have truncated HALF of these).
#   - context: 20 x (511 + ~120 user) tokens is ~13k, and the 866-token tail
#     ~17-20k, both inside --max-model-len 32768. A 4B on this card leaves
#     ~190k tokens of KV cache, against 12 x 13k = ~156k for full concurrency.
#   - model load + torch.compile + CUDA graph capture ~4 min (measured).
# --time 02:00:00 is >= 4x the central estimate.
#
# assert_pairs_share_prefix runs BEFORE anything is written: a prefix that
# quietly differed would turn a prompt difference into a measured causal gain,
# and the smoke already exercised it on 38 pairs.
#
# No pip install; this env has vLLM and nothing else.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Job 15738603 died in FlashInfer's JIT sampler for want of nvcc on the compute
# node; greedy decoding needs none of those kernels.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_sequor_s0_0_branch_arm.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --n-items 12 \
  --n-turns 20 \
  --max-new-tokens 1024 \
  --mode canonical \
  --out-dir outputs/sequor_s0_0_branch_arm
