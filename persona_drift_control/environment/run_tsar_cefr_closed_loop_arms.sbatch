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
# ===================== RULINGS, ALL SIGNED 2026-09-14 =========================
# The runner still raises on each in canonical mode rather than defaulting; the
# values below are the signed ones and the reason each was chosen is recorded so
# a later reader does not have to reconstruct it.
#
#   1. DECODING = sample, T = 0.7, top_p 0.8, top_k 20. These arms are
#      DETERMINISTIC policies, so unlike GPU-1 (whose seeds drew action
#      sequences) the seed axis has nowhere to live except decoding; greedy
#      with 3 seeds writes 3 byte-identical copies and a standard deviation of
#      exactly 0, which is a claim of precision that does not exist. The values
#      are Qwen3-4B's own non-thinking card defaults -- EXTERNAL, not tuned to
#      make the effect look better, the same discipline that put G-T1's 0.85 on
#      the published F1. Cost: this breaks byte-comparability with GPU-1 and the
#      tree, both greedy, which is what the ANCHOR job exists to repair.
#   2. STOP_ON_ARRIVAL = on_copy. Under `never` every arm spends exactly six
#      steps and `copy` returns the whole paragraph, so BOTH signed cost axes
#      (steps, tokens) go flat and plan 5.4's equal-cost comparison carries no
#      information. on_copy also matches greedy_reactive's own fourth rule.
#   3. BETA = 1.8333, from the trial 20 sources' human references: mean squared
#      level error 0.4474 / mean meaning loss 0.2441, i.e. plan 5.4's "official
#      50/50" read as equal expected contribution. The median gives 1.9282, so
#      the value is not sensitive to the statistic.
#      LAMBDA_COST = 0.1156 = 0.34^2, the published self-test RMSE of the TSAR
#      evaluator squared: do not spend an action chasing an improvement the
#      readout cannot resolve. Chosen over plan 5.4's literal "calibrate on the
#      token axis" because that would tune lambda until one arm's cost matched
#      another's; the two cost axes are reported instead.
#   4. MEANING THRESHOLD = 0.5436 (analyzer, not this job): the 5th percentile
#      of the trial human references' MeaningBERT-to-source, i.e. keep at least
#      as much meaning as the weakest human reference. Anchored on the
#      references, not on this model's output, so it cannot drift with the
#      effect. Failures score the SOURCE level (`source_level`), which keeps
#      every arm on the same denominator; `exclude` is reported beside it and
#      does not judge.
#   5. DP_PATH = level_prompt + trial_measured R. Its policy is a path of CEFR
#      LEVELS with multi-level jumps and the four control actions name no level;
#      restricting the DP to adjacent transitions does NOT rescue it, because
#      the strictly-decreasing path is then UNIQUE and dp_path IS fixed_ladder.
#      So tsar_cefr_actions gained `level_message`, which widens the level set of
#      a template that already existed; `saturation_message` keeps its A2/B1
#      contract and its output is byte-identical (regression test). R is
#      MODEL-DEPENDENT, so it is measured on the trial split with this model by
#      the published scheme rather than copied from the paper's Llama numbers --
#      the caption becomes "method reproduced from public code, R measured on
#      this model", not "parameters reproduced".
#      KNOWN RISK, COUNTED NOT REPAIRED: after min-max to [0,1] every R is
#      non-negative, so a SUM over hops rewards longer paths and the published
#      objective leans toward the ladder. `dp_degeneracy` in run_config.json
#      reports what share of items got the adjacent path; at 1.0 the arm is
#      fixed_ladder on this model and the analyzer's zero-variance guard fires.
# ==============================================================================
#
# SCALE: 199 items x 5 arms x 3 seeds x <= 6 steps = <= 17,910 generations (on_copy
# makes the real count lower; plan 6.1 assumes 60-70%), plus <= 800 for the reward
# probe on trial and 199 source readouts. The ANCHOR job adds 2,388 on top.
#
# RUNTIME ESTIMATE 0.8-1.5 h for THIS job (--time 4:00:00 = ~2.7x the
# pessimistic end); the anchor job adds ~0.2-0.3 h of its own.
# ASSUMPTIONS, all rates MEASURED ON GPU-2a (15897788), not carried over from
# any pre-run estimate -- the LEDGER records that GPU-2a's own generation guess
# was conservative and its readout guess overran the pessimistic end by 14%:
#   - GENERATION 0.0128 s/gen: GPU-2a's log sums to 14.4 min over 67,660
#     generations on an A100. 18,710 x 0.0128 = 4.0 min. Band 0.0128-0.03 ->
#     4-9 min.
#   - READOUT 0.111 s/text: GPU-2a wrote tree_raw.jsonl at 12:41 and tree.jsonl
#     at 14:46, so ~125 min over 67,859 texts through 3 ModernBERT-base
#     classifiers + MeaningBERT at batch 32. 18,909 x 0.111 = 35 min. THE BAND
#     IS WIDER HERE FOR A STRUCTURAL REASON: GPU-2a scored one batch of 67,660,
#     this job scores SIX batches of ~2,400 (the loop is closed, so step t+1
#     cannot be built before step t is read out). The probe models load once,
#     but per-batch overhead is paid six times -> 0.111-0.20 s/text -> 35-63 min.
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
# headroom. This job and the anchor are both METHOD, which cannot raise the
# instrument count.
#
# ARTIFACTS: outputs/tsar_cefr_s3_arm/ does not exist; the runner refuses to
# write into an existing directory. Nothing reads or rewrites GPU-1's rows, the
# tree, or the operator fit -- operator_fit.json is opened read-only.
#
# NO pip install here (item 6): the runner reaches src/ by file path, and
# nothing else of this project is queued.

set -euo pipefail

# ---- signed 2026-09-14; the guards below still refuse an unset value ---------
DECODING=sample
DECODING_TEMPERATURE=0.7
STOP_ON_ARRIVAL=on_copy
BETA=1.8333
LAMBDA_COST=0.1156
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

# PER-JOB COMPILE CACHES. vLLM's default cache root is ~/.cache/vllm on the shared
# NFS home, and its key is the model+config hash -- so two concurrent jobs on the
# same model write the same tree from two nodes and one of them gets ENOSTALE.
# That is exactly what killed 15943452 at 3m33s while 15943453 survived
# (node2087 vs node2093, cache writes at 21:26:17 and 21:26:25). Same failure class
# as the 2026-09-08 concurrent editable install (.claude/experiments.md item 6):
# two parallel jobs mutating one shared path. Scratch, scoped by job id, is private.
export VLLM_CACHE_ROOT=/scratch/hcao2/vllm_cache/job_${SLURM_JOB_ID:-local}
export TORCHINDUCTOR_CACHE_DIR=/scratch/hcao2/inductor_cache/job_${SLURM_JOB_ID:-local}
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_tsar_cefr_closed_loop_arms.py \
  --agent-model Qwen/Qwen3-4B \
  --split test \
  --arms one_shot fixed_ladder dp_path greedy_reactive koopman_mpc \
  --seeds 0 1 2 \
  --max-steps 6 \
  --decoding "$DECODING" "${TEMP_ARG[@]}" \
  --stop-on-arrival "$STOP_ON_ARRIVAL" \
  --top-p 0.8 \
  --top-k 20 \
  --dp-path-mapping level_prompt \
  --dp-path-reward trial_measured \
  --reward-probe-split trial \
  --reward-probe-depth 2 \
  --beta "$BETA" \
  --lambda-cost "$LAMBDA_COST" \
  --operator outputs/tsar_cefr_phase3/operator_fit.json \
  --readout-device 0 \
  --readout-batch-size 32 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --mode canonical \
  --out-dir outputs/tsar_cefr_s3_arm
