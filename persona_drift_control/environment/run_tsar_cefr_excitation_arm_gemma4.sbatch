#!/bin/bash
#SBATCH --job-name pdc-tsar-cefr-gemma4
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 4:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/tsar-cefr-gemma4-%j.out

# SECOND BACKBONE, tsar_cefr column of Table 1 (method: identification data).
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md item 1).
# Plan: docs/experiments/second_backbone_plan_2026-09-17.md
#
# WHAT IT DECIDES. Table 1 is currently eleven columns off ONE backbone, so
# "delay embedding helps / nonlinearity buys nothing" is not separable from
# "...on Qwen3-4B". This is the first of three behaviour-line arms that make it
# separable. It is deliberately first: cheapest (Qwen side 7m11) and the only
# line NAMING.md still marks active.
# It does NOT reopen any of this line's scientific questions -- G-T1, D-0,
# D-1(2), D-2, D-2.5 were all judged on the Qwen arm and are not re-judged here.
# What this job produces is the identification data for the six Table 1 rows.
#
# THE MODEL IS google/gemma-4-E4B-it (user ruling 2026-09-17). Availability was
# checked on the login node, not from memory:
#   - not gated; weights 16.02 GB already in /scratch/hcao2/hf_cache
#   - transformers 5.16.1 and vllm 0.28.0 both register Gemma4ForConditionalGeneration
#   - chat template handles multi-turn AND the system role, and accepts
#     (and ignores) enable_thinking= -- so chat_model.py's TypeError fallback
#     is never needed. This line does not use chat_model.py anyway (vLLM direct).
#
# "SAME 4B CLASS" IS THE EFFECTIVE-PARAMETER READING, NOT TOTAL PARAMS. E4B is
# 7.996 B total / 16 GB BF16, i.e. 2.1x Qwen3-4B's 7.6 GB. Gemma 4 has no dense
# 4B at all (dense starts at 12B). EVERY table that cites this arm must carry
# that sentence -- otherwise it reads as "a same-size backbone agrees", which is
# not what was run.
#
# NO CAP CALIBRATION JOB IS NEEDED HERE, AND THAT IS A PROPERTY OF THIS RUNNER,
# NOT AN ASSUMPTION. token_cap() is 1.5x the SOURCE length measured with the
# AGENT'S OWN tokenizer (line ~210: token_cap(len(tokenizer.encode(text)))), so
# it re-derives itself under Gemma's 262144-entry vocabulary. The arm then
# self-checks cap pressure PER ITEM against the pre-registered 5% threshold
# (cap_pressure / MAX_ITEM_CAP_HIT_SHARE) and reports items over it. The other
# two behaviour lines use FIXED caps tuned on Qwen and DO need calibration first
# -- that is why this arm is the one being submitted first.
# Precedent for taking this seriously: 15739196 (a cap tuned on 14B gave 8.1%
# parse failures on 4B) and 15756689 (18.6% truncation, forced a cap-2048 rerun).
#
# SCALE, identical to the Qwen arm so the columns are comparable: 100 test
# sources x 2 target levels x 3 seeds x 6 steps = 3,600 generations, + 40
# saturation probes on the trial split.
#
# RUNTIME ESTIMATE 0.4-1.0 h (--time 4:00:00 = 4x the pessimistic end).
# ASSUMPTIONS:
#   - Qwen arm 15850523 measured 00:07:11 end-to-end for exactly this schedule.
#   - Gemma E4B is 2.1x the weights and 1.7x the vocabulary, and its per-layer
#     embedding / MatFormer structure is NOT the dense-4B cost model, so the
#     Qwen rate is not transferable. Taking 2-4x that arm plus a larger model
#     load gives 15-35 min; the 1.0 h upper end is deliberate slack.
#   MEASURED on the login node (zero GPU), closing half of what was unmeasured:
#     the two tokenizers give near-identical budgets on these 100 paragraphs --
#     source tokens p50 103 -> 104, max 175 -> 176; derived cap p50 155 -> 156,
#     max 263 -> 264, total 16,191 -> 16,210 (+0.12%). So the cap does NOT
#     silently shrink or explode under the 262144-entry vocabulary, and cap
#     pressure is a question about GEMMA'S VERBOSITY ALONE, not about tokenizer
#     mismatch.
#   NOT MEASURED, and named as such: vLLM 0.28 throughput for gemma4 on this
#   node, and whether Gemma's own verbosity lands the per-item cap-hit share
#   under 5%. If it does not, the arm is NOT admitted and the fix is a named
#   --exclude, never a raised cap.
#   The runner writes trajectories_raw.jsonl BEFORE the readout pass, so a
#   classifier OOM costs only the CPU-cheap half.
#
# READOUT IS UNCHANGED AND THAT IS THE POINT: the CEFR classifiers and
# MeaningBERT are the same fixed models as the Qwen arm. Only the plant moves.
# --temperature 6.5 is the G-T1 value, NOT re-fitted for this backbone --
# re-fitting it here would make the two columns incomparable.
#
# QUOTA (.claude/experiments.md item 5): this job is METHOD.
#
# ARTIFACTS: outputs/tsar_cefr_gemma4_gpu1/ does not exist (dry-run checked; the
# runner refuses to write into an existing directory). The Qwen arm's
# outputs/tsar_cefr_gpu1/ is NOT touched -- outputs/ is append-only.
#
# NO pip install here (item 6): the runner reaches src/ by file path.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_tsar_cefr_excitation_arm.py \
  --agent-model google/gemma-4-E4B-it \
  --split test \
  --probe-split trial \
  --n-steps 6 \
  --seeds 0 1 2 \
  --temperature 6.5 \
  --readout-device 0 \
  --readout-batch-size 32 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --mode canonical \
  --out-dir outputs/tsar_cefr_gemma4_gpu1
