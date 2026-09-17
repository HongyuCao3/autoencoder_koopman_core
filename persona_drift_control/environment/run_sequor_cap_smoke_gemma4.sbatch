#!/bin/bash
#SBATCH --job-name pdc-sequor-cap-smoke-gemma4
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 1:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-cap-smoke-gemma4-%j.out

# SECOND BACKBONE, step 1 of the campaign's submission order: CAP CALIBRATION
# for the constraint line. INFRASTRUCTURE, not evidence -- --mode debug, whose
# own report stamps "[DEBUG: mode=debug, not evidence]". Nothing here may be
# reported; it exists to set --max-new-tokens for the real arm, which is the
# most expensive job in the whole campaign (Qwen side 4h22m + judging).
# Plan: docs/experiments/second_backbone_plan_2026-09-17.md §八 item 1.
#
# WHY. run_sequor_s1_pilot_arm.py takes a FIXED --max-new-tokens 2048, tuned on
# Qwen. It does compute cap_accounting against the pre-registered CAP_CRITERION
# = 0.05 (src/persona_drift/sequor_trajectory.py:318), so unlike the ergo line
# the failure would at least be visible -- but finding it in the 5-hour arm
# instead of here costs the arm. Precedent: 15756689 hit 18.6% truncation and
# forced a full rerun at cap 2048.
#
# WHAT IT DECIDES (gate lines, .claude/experiments.md):
#   passed  -> submit the constraint gemma-4 S1 arm at the calibrated cap,
#              then its independent 14B rejudge (the reportable readout).
#   failed  -> constraint LEAVES this group and the column is reported as
#              "this design cannot tell", per plan §六 death condition 3.
#              Calibrated ONCE, here; the cap is not raised a second time.
#   paper   -> Table 1 second-backbone group (new table), constraint column.
#
# SCALE. 2 items x 20 turns x 2 arms = 80 rows, against the Qwen arm's
# 40 items x 20 turns x 3 seeds x 4 arms = 9,600 rows (run_sequor_s1_arm.sbatch,
# COMPLETED 4h22m). Two arms, not one: zero_control and constant_remind bracket
# the length range, and cap pressure is a property of the longest arm, not the
# average one. Every other flag is copied verbatim from that sbatch.
#
# ONE FLAG DOES NOT MEAN ONE THING ACROSS BACKBONES, AND THIS IS THE LINE WHERE
# IT BITES: --decoding model_default reads each model's own generation_config,
# and they differ -- Qwen3-4B-Instruct-2507 is temp 0.7 / top_p 0.80 / top_k 20,
# gemma-4-E4B-it is temp 1.0 / top_p 0.95 / top_k 64. So on this line the
# backbone swap also swaps the decoding, and (unlike the core tasks) this group
# plans NO paired same-harness Qwen control here -- the published Qwen column is
# the control. Keeping the flag is what the harness convention says and is what
# this smoke runs; whether the real arm should instead pin Qwen's numbers (which
# needs a new CLI option, i.e. a change to a frozen harness) IS A USER RULING
# AND IS BEING ASKED BEFORE THE REAL ARM, not before this smoke. The cap is
# measured under the decoding the arm would actually use, so a ruling that
# changes the decoding invalidates this calibration and it must be re-measured.
#
# RUNTIME 5-20 min. Qwen did 9,600 rows in 4h22m => ~2.2 min for 80 rows; gemma
# measured 1.5x Qwen on the tsar_cefr arm (16018827); vLLM engine start on this
# node was ~3 min there. --time 1:00:00 is >=3x that.
#
# QUOTA: infrastructure (does not count, .claude/experiments.md item 5).
# ARTIFACTS: outputs/sequor_cap_smoke_gemma4/ is new; outputs/sequor_s1_arm/
# (the published Qwen column, incl. readout_independent_qwen3_14b.json) is not
# touched.
# NO pip install (item 6). This job DOES use vLLM, so it must not run beside
# another vLLM job -- 15943452 died in 3m33s on a shared compile cache. The ergo
# cap smoke submitted alongside it uses transformers, not vLLM.
# gemma-4-E4B-it max_position_embeddings = 131072 >= --max-model-len 49152
# (checked on the login node, not assumed).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_sequor_s1_pilot_arm.py \
  --agent-model google/gemma-4-E4B-it \
  --item-selection bank \
  --n-items 2 \
  --n-turns 20 \
  --seeds 0 \
  --arms zero_control constant_remind \
  --constraints-in-system \
  --decoding model_default \
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --mode debug \
  --out-dir outputs/sequor_cap_smoke_gemma4
