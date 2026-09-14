#!/bin/bash
#SBATCH --job-name pdc-tsar-cefr-gpu2
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 4:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/tsar-cefr-gpu2-%j.out

# tsar_cefr GPU-2: the FIVE CLOSED-LOOP ARMS of plan section 5.4 (method).
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md item 1), and it
# additionally refuses to start until five values are signed -- see RULINGS.
#
# WHAT IT DECIDES: Table 2's `tsar_cefr` column -- is Ours positive or zero.
# That is the line's paper-facing question; every gate so far only bought the
# right to ask it.
#
# WHY GPU-2a's TREE DOES NOT REPLACE THIS JOB. The tree enumerates every
# length-4 action sequence, so four arms can be read off it exactly -- at
# HORIZON 4, in ONE greedy realization. Table 2 is signed at <= 6 steps and at
# `mean +- std (n = 3 seeds)`. The tree matches neither axis. What the tree
# already settled stands: D-2.5 does not fire (+0.2241, CI [+0.1417, +0.3056]),
# so this job is allowed to exist.
#
# THE BOUND THIS JOB HAS TO CLEAR IS NOT THE OPEN-LOOP ARMS. D-2.5 measured the
# leave-one-source-out supremum over ALL bin-measurable closed-loop policies at
# RMSE 0.7752, against the hand-written `greedy_reactive` at 0.7882 -- a gap of
# 0.0130 -- while the best FITTED open-loop schedule sat at 0.9320. An Ours that
# beats the open-loop arms has cleared nothing that greedy_reactive had not
# already cleared without being fitted at all.
#
# ===================== RULINGS REQUIRED BEFORE SUBMISSION =====================
# The runner raises on each of these in canonical mode rather than defaulting,
# and the `: "${VAR:?}"` lines below make this script die in one second if it is
# submitted with any of them unset. Fill them in, do not delete the guards.
#
#   1. DECODING / DECODING_TEMPERATURE. These arms are DETERMINISTIC policies,
#      so unlike GPU-1 (whose seeds drew action sequences) the seed axis has
#      nowhere to live except decoding. Greedy decoding with 3 seeds writes 3
#      byte-identical copies and a standard deviation of exactly 0.
#      .claude/global.md forbids reporting a single-seed point estimate, so
#      either decoding is sampled here -- breaking byte-comparability with
#      GPU-1 and the tree, both greedy -- or Table 2 has no error bars.
#   2. STOP_ON_ARRIVAL. `never` generates a near-identity `copy` step and stays
#      comparable with the tree; `on_copy` ends the trajectory, which is what
#      plan 6.1's own cost estimate assumes. It changes both cost axes.
#   3. BETA / LAMBDA_COST. The MPC's meaning weight and per-action cost. Plan
#      5.4 fixes both on the TRIAL 20 sources (selection on trial, reporting on
#      test). They are not fitted here and not guessed here.
#   4. MEANING THRESHOLD (for the analyzer, not this job). kill_criterion 2.5
#      (3) records it as NOT YET SIGNED.
#   5. DP_PATH. Left out of ARMS below, and the runner refuses it either way.
#      Its policy is a path of CEFR LEVELS with multi-level jumps; this action
#      set is relative and names no level. Restricting the DP to adjacent
#      transitions does NOT rescue it -- the strictly-decreasing path becomes
#      UNIQUE, so dp_path IS fixed_ladder and the signed contrast
#      `koopman_mpc - dp_path` loses its reference. The only alternative is a
#      prompt naming an arbitrary level (tsar_cefr_actions restricts both of its
#      templates to the two target levels), i.e. a harness change.
# ==============================================================================
#
# SCALE with the four runnable arms: 199 items x 4 arms x 3 seeds x <= 6 steps
# = <= 14,328 generations and the same number of readout texts, plus 199 source
# readouts. With dp_path unblocked it is <= 17,910.
#
# RUNTIME ESTIMATE 0.6-1.3 h (--time 4:00:00 = ~3x the pessimistic end).
# ASSUMPTIONS, all rates MEASURED ON GPU-2a (15897788), not carried over from
# any pre-run estimate -- the LEDGER records that GPU-2a's own generation guess
# was conservative and its readout guess overran the pessimistic end by 14%:
#   - GENERATION 0.0128 s/gen: GPU-2a's log sums to 14.4 min over 67,660
#     generations on an A100. 14,328 x 0.0128 = 3.1 min. Band 0.0128-0.03 ->
#     3-7 min.
#   - READOUT 0.111 s/text: GPU-2a wrote tree_raw.jsonl at 12:41 and tree.jsonl
#     at 14:46, so ~125 min over 67,859 texts through 3 ModernBERT-base
#     classifiers + MeaningBERT at batch 32. 14,527 x 0.111 = 27 min. THE BAND
#     IS WIDER HERE FOR A STRUCTURAL REASON: GPU-2a scored one batch of 67,660,
#     this job scores SIX batches of ~2,400 (the loop is closed, so step t+1
#     cannot be built before step t is read out). The probe models load once,
#     but per-batch overhead is paid six times -> 0.111-0.20 s/text -> 27-48 min.
#   - MPC PLANNING, CPU: 340 operator rollouts per decision x 199 items x 3
#     seeds x 6 steps = 1.2M 4x4 updates in Python -> 1-5 min. NOT MEASURED.
#   - MODEL LOAD + ENGINE INIT ~3 min (GPU-2a: 12:23:58 start, 12:27:06 ready).
#   - PROMPT LENGTHS UNCHANGED: paragraph-level, context does NOT accumulate;
#     cap = 1.5x SOURCE tokens, GPU-1's rule, so rows stay comparable.
#   NOT MEASURED: vLLM 0.28 throughput at ~2,400-prompt batches on this node;
#   whether sampled decoding (if ruling 1 picks it) shifts output lengths and
#   therefore both the token axis and the cap-pressure rate.
#
# QUOTA (item 5): last ten jobs are instrument 3 / method 7 (15833850, 15815720,
# 15792580 instrument; the rest method). 3 <= 5, so the instrument axis has
# headroom. This job is METHOD.
#
# ARTIFACTS: outputs/tsar_cefr_s3_arm/ does not exist; the runner refuses to
# write into an existing directory. Nothing reads or rewrites GPU-1's rows, the
# tree, or the operator fit -- operator_fit.json is opened read-only.
#
# NO pip install here (item 6): the runner reaches src/ by file path, and
# nothing else of this project is queued.

set -euo pipefail

# ---- fill these in; the guards below refuse an unset value -------------------
# DECODING=sample
# DECODING_TEMPERATURE=0.7
# STOP_ON_ARRIVAL=never
# BETA=
# LAMBDA_COST=
# -----------------------------------------------------------------------------
: "${DECODING:?ruling 1 unset: greedy|sample -- the seed axis lives only in decoding}"
: "${STOP_ON_ARRIVAL:?ruling 2 unset: never|on_copy -- it moves both cost axes}"
: "${BETA:?ruling 3 unset: the MPC meaning weight, fixed on the trial 20 sources}"
: "${LAMBDA_COST:?ruling 3 unset: the MPC per-action cost, calibrated on the token axis}"
if [ "$DECODING" = "sample" ]; then
  : "${DECODING_TEMPERATURE:?--decoding sample needs a temperature}"
  TEMP_ARG=(--decoding-temperature "$DECODING_TEMPERATURE")
else
  TEMP_ARG=()
fi

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_tsar_cefr_closed_loop_arms.py \
  --agent-model Qwen/Qwen3-4B \
  --split test \
  --arms one_shot fixed_ladder greedy_reactive koopman_mpc \
  --seeds 0 1 2 \
  --max-steps 6 \
  --decoding "$DECODING" "${TEMP_ARG[@]}" \
  --stop-on-arrival "$STOP_ON_ARRIVAL" \
  --beta "$BETA" \
  --lambda-cost "$LAMBDA_COST" \
  --operator outputs/tsar_cefr_phase3/operator_fit.json \
  --readout-device 0 \
  --readout-batch-size 32 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --mode canonical \
  --out-dir outputs/tsar_cefr_s3_arm
