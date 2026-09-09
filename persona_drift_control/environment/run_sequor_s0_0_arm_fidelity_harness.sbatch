#!/bin/bash
#SBATCH --job-name pdc-sequor-s00-arm-fid
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 03:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s00-arm-fid-%j.out

# S0-0 BRANCH ARM, RE-RUN UNDER THE FIDELITY HARNESS (method).
# User ruling 2026-09-09 ("按照你的建议执行") after the fidelity arm's verdict.
#
# WHY THIS RE-RUN: job 15761176 showed the S0-0 arm's readout ceiling was OUR
# harness, not the task. Only both knobs together fix it -- constraint block in
# a system message AND the model's own sampling defaults gave upstream-comparable
# retention 0.42 at turn 1 / 0.08 by turn 20 (upstream: ~0.50 / <0.12 by turn
# 50), a paired -0.3292 [-0.5375, -0.1333] against the old harness, where each
# knob alone moved -0.033 and -0.075 with CIs spanning zero. On the graded
# readout the gates use, ceiling share falls 0.529 -> 0.200 with floor share
# 0.017, and K1's failing turns fall from three to one (t1 only).
#
# TWO CRITERION CHANGES, both signed off before this runs (screening doc
# section 10 item 5):
#   - K1's scope is t2..t20. Turn 1 states the constraints, admits no u=1
#     action and contributes ZERO pairs -- the same fact that made this arm 228
#     pairs and not 240. Turn 1's statistics are still computed and reported
#     under `per_turn_out_of_scope`, so the scope cannot hide a saturated
#     readout.
#   - 3 seeds instead of one greedy pass. Sampled decoding puts sampling noise
#     inside each counterfactual delta on top of the action: the prefix is still
#     byte-identical and the delta still unbiased, but its variance rises, which
#     raises the MDE that K2 and K3 are judged against. Pairs are keyed by seed,
#     so a pair is never formed across two trajectories.
#
# WHAT IT DECIDES: K1/K2/K3, recomputed. Fail -> the pre-registered closure
# stands, and this time it is not a harness artifact. Pass -> S1 opens.
#
# RUNTIME ESTIMATE 75-110 min. Measured inputs: the cap-2048 arm did 468
# generations in 15.7 min at 2.01 s/generation on an A100 80GB; the fidelity
# arm's `system_default` variant ran 2.10 s/generation with a 804-token median
# (against 524 greedy), so sampling with the block in the system message costs
# more tokens per response. 3 x 468 = 1404 generations at ~2.1-2.6 s => 50-60
# min on an 80GB card, up to ~90 min on a 40GB one; plus 4 min of load. Phase 2
# is 1404 x 3 = 4212 judge calls at a measured 0.10 s/call = ~7 min plus 4 min
# to load the 14B; phase 3 the same on the 4B at cap 1024. --time 03:00:00
# is >= 1.6x the pessimistic end.
#
# THREE PHASES, ONE JOB: the 4B generates, exits, then the 14B reporting judge
# scores, exits, then the 4B self judge scores. Two models will not co-reside on
# a 40GB card. Phase 3 exists because the in-loop judge's split-by-u gap is what
# plan section 2's known risk needs, and it is 8 minutes here against a whole
# job later. Each phase's artifacts survive a later phase failing.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Job 15738603 died in FlashInfer's JIT sampler for want of nvcc on the compute
# node. This arm samples, so it exercises the sampler for real.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

ARM=outputs/sequor_s0_0_branch_arm_fidelity_harness

echo "=== phase 1/3: branch arm, 3 seeds, system placement + model-default sampling ==="
python scripts/run_sequor_s0_0_branch_arm.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --n-items 12 \
  --n-turns 20 \
  --seeds 0 1 2 \
  --constraints-in-system \
  --decoding model_default \
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --mode canonical \
  --out-dir "$ARM"

echo "=== phase 2/3: reporting-judge readout (the gates read this one) ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --out-path "$ARM/readout_independent_qwen3_14b.json"

echo "=== phase 3/3: in-loop self-judge readout (selection signal, never reported) ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --judge-kind self \
  --max-new-tokens 1024 \
  --out-path "$ARM/readout_self_qwen3_4b_instruct_2507.json"
