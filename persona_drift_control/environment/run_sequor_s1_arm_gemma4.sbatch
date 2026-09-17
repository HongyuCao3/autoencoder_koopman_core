#!/bin/bash
#SBATCH --job-name pdc-sequor-s1-arm-gemma4
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 12:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s1-arm-gemma4-%j.out

# SECOND BACKBONE, constraint column of Table 1 (method: identification data).
# Plan: docs/experiments/second_backbone_plan_2026-09-17.md. This is the most
# expensive job in the campaign; everything cheap ran first on purpose.
#
# EVERY FLAG IS COPIED VERBATIM from run_sequor_s1_arm.sbatch -- the arm that
# produced the published Qwen column -- except --agent-model and --out-dir.
# Same 40 bank items, same 20 turns, same 3 seeds, same 4 schedule arms, same
# --constraints-in-system, same cap, same --max-model-len, --mode canonical.
#
# CAP 2048 IS MEASURED ON THIS BACKBONE, NOT INHERITED: the cap smoke 16019471
# ran gemma over 80 rows on the two bracketing arms (zero_control and
# constant_remind) and reported "cap: 0/80 (0.0%) overall; items over the 5%
# criterion: none" against the pre-registered CAP_CRITERION = 0.05. Qwen's cap
# therefore carries over here. (It did NOT carry over on gsm8k_sharded -- that
# line's smoke 16019470 truncated an item at 0.200 and is being re-measured.)
#
# THE DECODING MOVES WITH THE BACKBONE, BY USER RULING (2026-09-17), AND THE
# TABLE NOTE MUST SAY SO. --decoding model_default reads each model's own
# generation_config: Qwen3-4B-Instruct-2507 is temperature 0.7 / top_p 0.80 /
# top_k 20, gemma-4-E4B-it is 1.0 / 0.95 / 64. This line runs NO paired
# same-harness Qwen control -- the published Qwen column is the control -- so
# what this column measures is "a different backbone at its own default
# decoding", not "the same decoding on different weights". The user was asked
# this before submission and ruled to keep the harness convention. Any sentence
# citing this column carries that caveat at the same site.
#
# RUNTIME ESTIMATE 5.5-8.5 h (--time 12:00:00). ASSUMPTIONS:
#   - the Qwen arm measured 4h22m for exactly this configuration (9,600 rows);
#   - gemma measured 1.5x Qwen end-to-end on the tsar_cefr arm (16018827,
#     10m48 vs 7m11) under the same vLLM version and node class;
#   - 4h22m x 1.5 = 6.5 h, and the band is +/-30% around that.
#   NOT a basis: the cap smoke's 14m12 for 80 rows. That run had 2 concurrent
#   trajectories and is dominated by engine start and by having no batch to
#   fill; the real arm runs 160 trajectories per turn. Quoting it would give a
#   22-hour number that means nothing.
#
# THE READOUT IS A SEPARATE JOB AND MUST BE (run_sequor_s1_score.sbatch's own
# reasoning): a judging failure must not cost the generation. Only the
# INDEPENDENT Qwen3-14B readout is needed here -- that is the reportable one
# under .claude/global.md, and it stays Qwen3-14B precisely because it must be
# independent of the agent. The in-loop self readout is a selection signal this
# arm has no selection to make with, so it is not run.
#
# QUOTA: method.
# ARTIFACTS: outputs/sequor_s1_arm_gemma4/ is new; outputs/sequor_s1_arm/
# (the published Qwen column and its readout_independent_qwen3_14b.json) is
# untouched.
# NO pip install (item 6). Uses vLLM, so it must not run beside another vLLM
# job (15943452 died in 3m33s on a shared compile cache) -- checked at submit.

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
  --n-items 40 \
  --n-turns 20 \
  --seeds 0 1 2 \
  --arms zero_control constant_remind bernoulli antithetic \
  --constraints-in-system \
  --decoding model_default \
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --mode canonical \
  --out-dir outputs/sequor_s1_arm_gemma4
