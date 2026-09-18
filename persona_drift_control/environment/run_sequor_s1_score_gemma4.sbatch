#!/bin/bash
#SBATCH --job-name pdc-sequor-s1-score-gemma4
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 04:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s1-score-gemma4-%j.out

# THE SECOND BACKBONE'S constraint READOUT (instrument) -- judging only, no
# generation. Reads the rows 16021868 landed in outputs/sequor_s1_arm_gemma4.
# Plan: docs/experiments/second_backbone_plan_2026-09-17.md section 7.
#
# ONE READOUT, NOT TWO, and that is the whole difference from
# run_sequor_s1_score.sbatch. The self judge exists to be S3's in-loop
# SELECTION signal; this group runs no S3 and no controller, so a self readout
# here would be a number with no consumer and, per .claude/global.md, no
# licence to be reported either. The independent Qwen3-14B readout is the only
# one Table 1 may read.
#
# THE JUDGE DOES NOT MOVE WITH THE BACKBONE. Same judge model, same cap 512,
# same script as the published Qwen column's 15772957. Only the text being
# judged was written by a different agent. Changing the judge here would
# confound "another backbone" with "another readout" -- which is precisely the
# comparison this group is built to avoid.
#
# RUNTIME ESTIMATE 0.8-1.3 h. ASSUMPTIONS: 9,600 rows x 3 constraints = 28,800
# calls, the same count as 15772957, which measured 0.1115 s/call on the 14B at
# cap 512 = 53.5 min, plus ~4 min to load. The gemma arm's responses are
# SHORTER than the Qwen arm's (median 523 vs 711 output tokens, arm_report.json
# both sides), so the judge's prompts are shorter and this should not exceed
# the Qwen timing; the band is that estimate +/- 50%. --time 04:00:00 is ~3x
# the pessimistic end. UNMEASURED: nothing -- same judge, same cap, same task,
# and the only moving part is prompt length, which moved down.
#
# QUOTA: instrument (rejudge/readout).
# ARTIFACTS: outputs/sequor_s1_arm_gemma4/readout_independent_qwen3_14b.json is
# new (checked before submission); the script refuses to overwrite it anyway,
# and the arm's trajectories.jsonl is read-only here. The published Qwen
# readout in outputs/sequor_s1_arm/ is not touched.
# NO pip install (item 6 of .claude/experiments.md).
# USES vLLM, so it is submitted in the same --dependency chain as the core arm
# pair rather than beside them.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

ARM=outputs/sequor_s1_arm_gemma4

echo "=== reporting-judge readout (the only one any number may come from) ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --out-path "$ARM/readout_independent_qwen3_14b.json"
