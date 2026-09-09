#!/bin/bash
#SBATCH --job-name pdc-sequor-vllm-consistency
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 00:40:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-vllm-consistency-%j.out

# BACKEND CONSISTENCY CHECK (infrastructure, not evidence -- --limit writes
# mode="debug", and plan section 8 forbids citing debug artifacts).
#
# WHAT IT BUYS: which backend the S0 calibration is allowed to run on. Job
# 15719117 measured the judge at 7.42 s/row on HF transformers -> 4.12 h per
# candidate; vLLM batches the same 2000 rows into ~half an hour. But swapping
# the serving stack under a measurement instrument is the move that cost the
# ERGO line 18 jobs re-answering one question, so the vLLM stack scores the
# SAME 24 gold rows the HF smoke run scored (--limit 24 takes a deterministic
# stratified slice; tests/test_sequor_judge_calibration_vllm.py asserts the
# selection is byte-identical to the smoke report's row list) and
# --compare-to reports the verdicts row by row.
#
# ACCEPTANCE (engineering, pre-registered, NOT a scientific gate):
#   >= 22/24 verdicts identical (<= 2 flips)  -> run S0 full on vLLM.
#   <= 21/24                                  -> vLLM is a different
#     instrument: do NOT report numbers from it. Fall back to the HF path
#     (4.12 h/candidate) and treat the gap as an engineering bug (chat
#     template / enable_thinking pass-through), never as a finding.
# What 24 rows CANNOT buy: equivalence within a point or two of accuracy.
# It detects gross template/parse breakage. Both backends decode greedily
# from identical prompt bytes, so anything beyond a couple of numeric
# tie-break flips means the stacks differ.
#
# G-S0-1/G-S0-2 are absolute thresholds evaluated on the 2000-row run itself,
# so backend equivalence is not what makes the gate valid -- it is what makes
# the numbers comparable to upstream's calibration and to the HF smoke.
#
# RUNTIME ESTIMATE 5-10 min. Assumptions: A100-PCIE-40GB (what job 15719117
# landed on), Qwen3-14B bf16 = 28 GB of weights loaded from /scratch, 24 rows
# in a single batch, judge output median ~185 tokens (measured: 741 chars
# median, 1351 max on the smoke run), cap 512, greedy, thinking off. The
# dominant term is model load + CUDA graph capture, not generation.
# --time 00:40:00 is >= 4x the estimate.
#
# No pip install of any kind: this env has vLLM and nothing else, and the
# runner loads the shared prompt/parser/metric modules by file path
# (scripts/calibrate_sequor_judge_vllm.py load_shared_module). Nothing here
# touches /scratch/hcao2/envs/persona_drift_pilot, so it cannot reproduce the
# 2026-09-08 concurrent-editable-install race.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 2026-09-09, job 15738603 FAILED here after loading weights and capturing CUDA
# graphs: vLLM's sampler routes to FlashInfer, whose sampling kernels JIT-compile
# on the first token and need nvcc / CUDA_HOME, neither of which exists on the
# compute node (the env's torch ships its CUDA libs as wheels, not a toolkit).
# Greedy decoding does not need those kernels at all. `module load cuda/13.0.2`
# would also supply nvcc, but it puts a second CUDA runtime on LD_LIBRARY_PATH
# ahead of the wheels this env was built against -- a numerical-stack change
# under an instrument, which is what the separate env exists to prevent.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/calibrate_sequor_judge_vllm.py \
  --judge-model Qwen/Qwen3-14B \
  --max-new-tokens 512 \
  --limit 24 \
  --compare-to outputs/sequor_s0_smoke/judge_calibration_qwen3_14b_smoke.json \
  --out-path outputs/sequor_s0_vllm_consistency/judge_calibration_qwen3_14b_vllm_smoke24.json
