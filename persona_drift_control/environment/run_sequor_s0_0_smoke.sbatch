#!/bin/bash
#SBATCH --job-name pdc-sequor-s00-smoke
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 01:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s00-smoke-%j.out

# S0-0 BRANCH ARM SMOKE (infrastructure, not evidence -- --mode debug, and
# plan section 8 forbids citing debug artifacts; the gate analyzer REFUSES a
# readout whose arm ran in debug mode, so these numbers cannot leak into K1/K2/K3).
#
# WHAT IT BUYS, the three numbers the S0-0 runtime estimate is missing
# (plan section 9 names them: "the first action is a 2-trajectory smoke test"):
#   1. the agent's response length on this task -- upstream never reported its
#      generation settings, and the readout becomes partly a truncation
#      measurement if the cap bites. The runner prints the share hitting the cap
#      and flags > 5%.
#   2. seconds per generation at T=20 with the context growing to ~20 turns,
#      which is what turns "S0-0 is a small arm" into a number.
#   3. whether a T=20 dialogue fits --max-model-len 32768 at all.
# It also exercises the pair machinery end to end: assert_pairs_share_prefix
# runs before anything is written, so a prefix bug fails here at 2 items
# rather than after the full arm.
#
# 2 items x 20 turns = 20 base + 19 counterfactual generations per item
# = 78 generations. NOT the pre-registered arm (N=12, 228 pairs) -- that runs
# only after S0 has named the two judges, and its own sbatch is written then.
#
# RUNTIME ESTIMATE 10-25 min. Assumptions: A100-PCIE-40GB;
# Qwen3-4B-Instruct-2507 bf16 (~8 GB weights; the 14B judge run measured 6.14
# GiB / 40k tokens of KV cache on this card, so a 4B leaves far more);
# 78 generations capped at 1024 tokens = up to ~80k output tokens at roughly
# 1-2k tok/s aggregate = 1-2 min of decode, but the batches are only 2 wide, so
# per-token latency dominates over throughput: ~15-25 s per generation pair-turn
# is the pessimistic case. Model load + torch.compile + CUDA graph capture cost
# ~4 min (measured on job 15738603, a 14B). --time 01:00:00 is >= 2.5x the
# pessimistic estimate.
#
# The 2 items are the first two of the 16 dialogues whose k=3 constraints are
# ALL inside the judge-calibration gold set (screening section 9.2), i.e. the
# same selection rule the canonical arm uses -- so the timings transfer.
#
# No pip install: this env has vLLM and nothing else.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# See run_sequor_judge_vllm_consistency.sbatch: job 15738603 died in FlashInfer's
# JIT sampler for want of nvcc on the compute node. Greedy decoding needs none
# of it.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_sequor_s0_0_branch_arm.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --n-items 2 \
  --n-turns 20 \
  --max-new-tokens 1024 \
  --mode debug \
  --out-dir outputs/sequor_s0_0_smoke
