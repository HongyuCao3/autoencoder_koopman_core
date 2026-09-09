#!/bin/bash
#SBATCH --job-name pdc-sequor-s00-arm-c2048
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 02:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s00-arm-cap2048-%j.out

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
# RE-RUN AT cap 2048 (user ruling 2026-09-09, option a). The 1024-token run
# (job 15756689) FAILED its own pre-registered criterion: 87/468 responses
# truncated (18.6%), and 75 of those 87 sat on two items whose constraints ask
# for length -- `tuple_294_1` "Write a longer dialog" at 100% and
# `tuple_245_1` "Write creatively as a story" at 92.3%, against <=12.8% for the
# other ten. A truncated response can fail a constraint the model would have
# satisfied, so that readout is partly a measurement of our own token budget.
#
# PRE-REGISTERED, before this runs: the criterion is PER ITEM at 5%, and if any
# item still binds, that is REPORTED and handed back -- no cap raised again on
# the agent's own initiative, no item dropped. The runner now writes
# `token_cap_by_item` and `items_over_cap_criterion`, and
# analyze_sequor_s0_0_gates.py REFUSES the arm until each offending item is
# either fixed or named in --exclude-item, which lands in the gate report next
# to the numbers it changed. Excluding items because their constraints demand
# length would bias the item set toward constraints that are easy to keep --
# that is a finding about the design, not a detail to tidy away.
#
# --max-model-len 49152 follows from the cap: 20 turns x (2048 + ~120 user)
# tokens is ~43k, and 32768 would silently drop the early turns -- which is
# where the constraints were stated, i.e. the one part of the context that
# cannot be lost.
#
# RUNTIME ESTIMATE 20-45 min. Measured inputs: the 1024 run did 468
# generations in 9.3 min at 1.19 s/generation, but on an A100 80GB (KV cache
# 62.6 GiB, 455,888 tokens, 13.9x concurrency at 32k/request); the smoke got a
# 40GB card and 3.14 s/generation at batch width 2. `--gpus a100:1` gives
# either card, so the spread is real: at 49k tokens/request an 80GB node holds
# ~9.3x concurrency and a 40GB node ~3.9x, and only the previously-truncated
# ~19% of responses generate more tokens than before. --time 02:00:00 is >= 2.7x
# the pessimistic end.
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
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --mode canonical \
  --out-dir outputs/sequor_s0_0_branch_arm_cap2048
