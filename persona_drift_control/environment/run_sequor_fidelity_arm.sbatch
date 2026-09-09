#!/bin/bash
#SBATCH --job-name pdc-sequor-fidelity
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 02:30:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-fidelity-%j.out

# UPSTREAM-FIDELITY ARM (instrument). User ruling 2026-09-09: 开始测试.
#
# WHY: the S0-0 arm's readout sits 1.7x above the benchmark it came from --
# 0.833 of turns with all k=3 constraints held at turn 1, against upstream's
# ~0.50 for this same model id, decaying only to 0.583 by turn 20 where
# upstream reaches <0.12 by turn 50. That ceiling is why 56% of the arm's 228
# pairs had no headroom (delta_y +0.0026 at y_prev=1.00 vs +0.125..+0.20
# wherever headroom existed), and therefore why K1 failed. Until this is
# settled, "close the line" and "our harness is too easy" are the same
# evidence.
#
# WHAT IT DECIDES: whether either knob we chose ourselves is the cause.
#   placement: constraint block in USER TURN 1 (what S0-0 did, INFERRED from
#              the vendored field and never checked against upstream code)
#              vs a SYSTEM message.
#   decoding:  GREEDY (what S0-0 did) vs the model's OWN DEFAULTS, which is
#              what upstream reported. The defaults are read from the model's
#              generation_config at runtime (T=0.7 / top_p=0.8 / top_k=20 for
#              Qwen3-4B-Instruct-2507) rather than typed here.
# 2x2, zero-control, the SAME 12 items as S0-0 so every variant is paired
# item-for-item with it and only the harness differs.
#
# PRE-REGISTERED (before submission; ledger + screening section 10): a variant
# closes the gap if its mean upstream-comparable retention over t1..t20 is at
# least 0.10 below the turn1_greedy baseline with an item-paired bootstrap CI
# excluding 0, AND its turn-1 value is <= 0.60.
#   pass -> re-run the S0-0 branch arm under that variant, recompute K1/K2/K3.
#   no variant passes -> the readout has no range on this task with this model
#     and the pre-registered closure of the line stands.
# Verdict comes from scripts/analyze_sequor_fidelity.py (CPU, run after), not
# from this job.
#
# ONE JOB, TWO PHASES, on purpose: generation with the 4B, then scoring with
# the 14B reporting judge in a separate process after the first releases the
# GPU. Two models will not co-reside on a 40GB card, and `--gpus a100:1` gives
# either card. It is also what keeps this inside the instrument quota -- 5 of
# the last 10 jobs are already instruments and 2 more would trip the stop
# rule (docs/LEDGER.md). Both facts are stated because the second one alone
# would be a reason to distrust the arrangement.
#
# RUNTIME ESTIMATE 45-70 min. Measured inputs: the S0-0 arm did 468
# generations in 15.7 min at 2.01 s/generation with cap 2048 on an A100 80GB;
# this is 4 x 240 = 960 generations of the same shape, so ~32 min if the card
# matches and up to ~55 min on a 40GB card, plus ~4 min of model load. Phase 2
# is 960 x 3 = 2880 judge calls at the measured 0.10 s/call = ~5 min plus ~4
# min to load the 14B. The sampled variants may run longer than greedy -- that
# is unmeasured. --time 02:30:00 is >= 2x the pessimistic end.
#
# Phase 2 runs only if phase 1 succeeded (set -e), and phase 1's artifacts
# survive a phase-2 failure.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Job 15738603 died in FlashInfer's JIT sampler for want of nvcc on the compute
# node. Note this arm SAMPLES in two of its four variants, so unlike the
# greedy-only jobs it genuinely exercises the sampler -- vLLM's own PyTorch
# path handles it.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

echo "=== phase 1/2: generation, 4 variants, one model load ==="
python scripts/run_sequor_fidelity_arm.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --n-items 12 \
  --n-turns 20 \
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --mode canonical \
  --out-dir outputs/sequor_fidelity_2x2

echo "=== phase 2/2: reporting-judge readout over all four variants ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir outputs/sequor_fidelity_2x2 \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --out-path outputs/sequor_fidelity_2x2/readout_independent_qwen3_14b.json
