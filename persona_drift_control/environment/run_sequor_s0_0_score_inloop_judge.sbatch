#!/bin/bash
#SBATCH --job-name pdc-sequor-s00-score-inloop
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 01:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s00-score-inloop-%j.out

# S0-0 READOUT, IN-LOOP (SELF) JUDGE (instrument).
# The same responses, scored by the agent's own model. These verdicts are a
# SELECTION signal: the closed loop feeds back on them in S3, and no reported
# number ever comes from them (.claude/global.md -> 报告口径; that file's named
# exception covers `defense` only). The artifact records judge_kind=self, and
# the gate analyzer refuses a self-judged readout where it wants the
# independent one.
#
# WHAT IT DECIDES, two things:
#   1. Whether the in-loop state is usable at k=3. The S0 calibration (job
#      15739196) found this model's verdict prose runs to a p90 of 463 tokens
#      against the 14B's 194, so at --max-new-tokens 512 it left 8.1% of calls
#      unparsed -- and `graded_readout` nulls a whole turn if ANY of its k
#      constraints is unparsed (so `y` can never silently change scale), which
#      at k=3 is ~22% of turns with no state for a controller to feed back on.
#      --max-new-tokens 1024 HERE is the fix, and this job is where it is
#      measured: no separate instrument job was opened for it (screening doc
#      section 10 item 4, user sign-off 2026-09-09). If the unusable-row share
#      is still material, report it before S3 is designed -- do not lower the
#      readout's None policy to make the number look better.
#   2. How far the selection signal sits from the reporting one, split by `u`
#      (analyze_sequor_s0_0_gates.py --self-readout). A judge that is more
#      generous on reminded rows would inflate `B` by an instrument artifact;
#      the gold set has no `u`, so this is the first look at plan section 2's
#      known risk -- weaker than the strong-judge sample, and not a substitute.
#
# RUNTIME ESTIMATE 6-15 min. Assumptions: A100-PCIE-40GB;
# Qwen3-4B-Instruct-2507 bf16 (~8 GB weights); 468 rows x k=3 = 1404 calls;
# 0.099 s/call measured on 2000 gold rows at cap 512 (job 15739196), and the
# cap here is doubled, so budget ~0.15-0.20 s/call = 4-5 min of batch plus
# ~3 min of load. --time 01:00:00 is >= 4x the estimate.
#
# Run AFTER (not beside) the reporting-judge scoring job: one GPU, one vLLM.

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
  --arm-dir outputs/sequor_s0_0_branch_arm \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --judge-kind self \
  --max-new-tokens 1024 \
  --out-path outputs/sequor_s0_0_branch_arm/readout_self_qwen3_4b_instruct_2507.json
