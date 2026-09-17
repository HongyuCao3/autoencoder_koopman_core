#!/bin/bash
#SBATCH --job-name pdc-ergoUP-excite-gemma4
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 6:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoUP-excite-gemma4-%j.out

# SECOND BACKBONE, gsm8k_sharded column of Table 1 (method: identification data).
# Plan: docs/experiments/second_backbone_plan_2026-09-17.md
#
# EVERY FLAG IS COPIED VERBATIM from run_ergo_upB_random_excite.sbatch -- the arm
# that produced the published Qwen column (15696226, COMPLETED 43m, 666 rows) --
# except --agent-model, --output-dir, and the cap, which is the one thing this
# line could not inherit.
#
# THE CAP IS 2048 AND IT IS MEASURED ON THIS BACKBONE, IN TWO PASSES:
#   pass 1 (16019470, cap 512 = Qwen's value): one reply landed exactly on 512,
#     item ergo_GSM8K_214 at a per-item truncation share of 0.200 against the
#     pre-registered 0.05 => Qwen's cap does NOT carry over.
#   pass 2 (16021852, cap 2048): 0/20 truncated, worst item 0.000, and the reply
#     that pass 1 clipped turns out to want 779 tokens. Median 87 and p90 473 are
#     unchanged between passes, so only the tail moved.
# 2048 is submitted because it is the condition that was measured; 1024 would be
# interpolation between two measurements, and a cap costs nothing when the reply
# ends at EOS. This is the ONE calibration the plan allows: if the arm still
# exceeds 5% per item, gsm8k_sharded leaves the group (plan §六 death condition
# 3) and is reported as "this design cannot tell". The cap is not raised again.
#
# THE 30% PARSE-FAILURE RATE IS NOT A CAP PROBLEM AND IS NOT BEING FIXED -- and
# the two passes now say so as a CONTROLLED COMPARISON, not as a reading of
# samples: judge_parse_failure_rate is 0.300 at cap 512 and 0.300 at cap 2048,
# identical while truncation went 0.200 -> 0.000. Against the Qwen arm's 0.018,
# what it measures is that gemma writes "Current answer: None" / "Unable to
# determine" on early shards where Qwen guesses a number (refusal rates are
# comparable, 0.15 vs 0.167). The harness already maps an unparseable answer to
# y_task_success = 0.0 -- checked row by row on both arms, no row dropped, no
# NaN -- which is what the metric means. Touching the parser would be changing
# the readout of an already-published column.
#
# RUNTIME 1.5-2.5 h (--time 6:00:00, copied from the Qwen arm). ASSUMPTIONS:
#   - this runner is SEQUENTIAL per trajectory (transformers, no batching): the
#     smoke logs show ~27 s per trajectory, i.e. ~10 s per row at cap 2048;
#   - 666 rows x 10 s = 111 min, plus ~3.5 min of model load (both smokes).
#   - So gemma is ~2.6x the Qwen arm here (43m for the same 666 rows), NOT the
#     1.5x measured on the vLLM tsar_cefr arm. The difference is the longer
#     replies the new cap admits, and it is why this number is derived from this
#     line's own smokes instead of from that ratio.
#
# QUOTA: method.
# ARTIFACTS: outputs/ergo_upB_random_excite_gemma4/ is new; the Qwen arm's
# outputs/ergo_upB_random_excite/ is untouched.
# NO pip install (item 6): the runner puts src/ on sys.path itself. It uses
# transformers, NOT vLLM, so it may run beside the constraint arm (16021868)
# without contending for the vLLM compile cache that killed 15943452.

set -euo pipefail
module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
nvidia-smi -L || true

python scripts/run_ergo_math_screening.py \
  --agent-model google/gemma-4-E4B-it \
  --device cuda \
  --output-dir outputs/ergo_upB_random_excite_gemma4 \
  --num-items 60 --item-rng-seed 2 \
  --seeds 0 1 \
  --controller random_excite --random-excite-p 0.5 \
  --reset-mode append \
  --prompt-profile upstream \
  --agent-max-new-tokens 2048
