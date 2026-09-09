#!/bin/bash
#SBATCH --job-name pdc-sequor-s0-inloop-judge
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 01:30:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s0-inloop-judge-%j.out

# S0 JUDGE CALIBRATION, IN-LOOP JUDGE (instrument).
# docs/experiments/constraint_retention_plan.md section 3: the in-loop judge is
# the SAME served model as the agent, so its verdicts are self-judged and are a
# SELECTION signal only -- the controller feeds back on them, and no reported
# number ever comes from them (.claude/global.md -> 报告口径; that file's named
# exception covers `defense` only and explicitly does not spill to this line).
#
# THE MODEL ID HERE IS THE AGENT'S MODEL. Qwen3-4B-Instruct-2507 is what
# upstream SEQUOR ran on the tuples/3 split, and plan section 1's range prior
# (tuples ~50% at turn 1 -> <12% by turn 50) is read off that model's curve --
# the only evidence this task has a usable range at all. Changing this id
# changes the system under study and puts that prior out of scope.
#
# WHAT IT DECIDES: whether the closed loop can feed back on the agent's own
# verdicts at all. Being weaker than the reporting judge is expected and fine
# (noise, not bias, as long as the error rate is not correlated with `u` --
# plan section 2 keeps that as a separate sampling check, which THIS run
# cannot answer because the gold set contains no `u`).
#
# ACCEPTANCE (pre-registered): balanced accuracy >= 0.65 AND at least +10
# points over the G-S0-0 paired length floor (64.8%).
#   pass -> this is the in-loop readout for the S0-0 screening arm and S3.
#   fail -> the in-loop state would be a response-LENGTH proxy, not constraint
#     satisfaction, and a controller feeding back on it would be steering on
#     verbosity. Swap the instrument: run the reporting judge in-loop as well
#     (slower S3, no self-judging), do NOT lower this threshold.
#
# RUNTIME ESTIMATE 8-20 min. Assumptions: A100-PCIE-40GB; Qwen3-4B-Instruct-2507
# bf16 (~8 GB weights, so ~28 GB of KV cache at --gpu-memory-utilization 0.90
# and far more concurrency than the 14B run); same 2000 prompts (~1.3M prefill
# tokens) and the same 512-token cap; a 4B model is ~3.5x cheaper per token
# than the 14B, whose central estimate is 15-30 min. Output length on this
# model is NOT measured -- the smoke run (job 15719117) used the 14B, so if
# this model is chattier before its verdict the decode term grows.
# --time 01:30:00 is >= 4x the estimate.
#
# --compare-to is absent on purpose: no HF-backend report exists for this
# model, and the backend consistency evidence comes from the 14B check.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/calibrate_sequor_judge_vllm.py \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --max-new-tokens 512 \
  --out-path outputs/sequor_s0_inloop_judge_qwen3_4b_instruct_2507/judge_calibration_qwen3_4b_instruct_2507_vllm.json
