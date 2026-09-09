#!/bin/bash
#SBATCH --job-name pdc-sequor-s00-score-report
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 01:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s00-score-report-%j.out

# S0-0 READOUT, REPORTING JUDGE (instrument).
# Applies y = (satisfied constraints)/k to the canonical branch arm with the
# judge S0 chose: Qwen3-14B, independent of the agent (agreement 0.8944,
# kappa 0.7886, +24.6 points over the paired length floor; job 15739195).
#
# WHAT IT DECIDES: all three S0-0 gates. Per the screening doc's section 10
# item 3 (user sign-off 2026-09-09), K1/K2/K3 are computed on THIS readout,
# not on the self-judged one -- K1 asks whether the reportable readout has any
# range at all, which is exactly how the `defense` line died, and asking a
# selection signal that question would repeat it.
#
# RUNTIME ESTIMATE 8-15 min. Assumptions: A100-PCIE-40GB; 12 items x
# (20 base + 19 counterfactual) = 468 rows x k=3 constraints = 1404 judge
# calls; 0.185 s/call measured on 2000 gold rows with this model (job
# 15739195), so ~4-5 min of batch; model load + torch.compile + CUDA graph
# capture ~4 min (measured, same job). The prompts here are agent responses
# rather than gold ones, so their length is what the smoke run measures.
# --time 01:00:00 is >= 4x the estimate.
#
# --max-new-tokens stays at the default 512: measured sufficient for THIS
# model (median 124 output tokens, p90 194, 12/2000 truncated). The 4B needs
# more, and its own scoring job passes 1024.
#
# DO NOT run this concurrently with the self-judge scoring job on the same arm
# directory. Different --out-path, but two vLLM processes on one GPU will not
# both fit, and a requeued job is indistinguishable from a dead one here.

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

python scripts/score_sequor_trajectories.py \
  --arm-dir outputs/sequor_s0_0_branch_arm_cap2048 \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --out-path outputs/sequor_s0_0_branch_arm_cap2048/readout_independent_qwen3_14b.json
