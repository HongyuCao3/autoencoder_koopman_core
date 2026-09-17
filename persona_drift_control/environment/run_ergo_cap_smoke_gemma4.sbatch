#!/bin/bash
#SBATCH --job-name pdc-ergo-cap-smoke-gemma4
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 1:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergo-cap-smoke-gemma4-%j.out

# SECOND BACKBONE, step 1 of the campaign's submission order: CAP CALIBRATION
# for the gsm8k_sharded line. INFRASTRUCTURE, not evidence -- nothing from this
# job may be reported; it exists to set --agent-max-new-tokens for the real arm.
# Plan: docs/experiments/second_backbone_plan_2026-09-17.md §八 item 1.
#
# WHY THIS LINE NEEDS IT AND tsar_cefr DID NOT. run_tsar_cefr_excitation_arm.py
# derives its cap per item from the agent's own tokenizer, so it re-derives
# itself under a new vocabulary (16018827 came back with 0 items over cap).
# run_ergo_math_screening.py instead takes a FIXED --agent-max-new-tokens,
# default 512, tuned on Qwen -- and this runner has NO cap accounting at all
# (grep: no truncation share is computed or written). That combination is
# exactly what produced 15739196 (a 14B-tuned cap gave 8.1% parse failures on
# 4B) and 15756689 (18.6% truncation, forced a cap-2048 rerun).
#
# WHAT IT DECIDES (gate lines, .claude/experiments.md):
#   passed  -> submit the gsm8k_sharded gemma-4 random-excitation arm (the
#              mainline identification job) at the cap this smoke measures.
#   failed  -> gsm8k_sharded LEAVES this group and the column is reported as
#              "this design cannot tell", per plan §六 death condition 3.
#              The cap is calibrated ONCE, here; it is not raised a second time.
#   paper   -> Table 1 second-backbone group (new table), gsm8k_sharded column.
#
# SCALE. 4 items x 1 seed ~= 22 rows, against the Qwen arm's 60 items x 2 seeds
# = 666 rows (15696226, COMPLETED 43m). Every other flag is copied verbatim from
# environment/run_ergo_upB_random_excite.sbatch -- the arm that produced the
# published Qwen column -- so what the smoke measures is the cap under the real
# protocol, not under a simplified one.
#
# DECODING DOES NOT DRIFT ON THIS LINE: chat_model.GenerationConfig fixes
# temperature 0.7 / top_p 0.95 independently of the model, so only the weights
# move. (The constraint line is NOT like this -- see its own smoke.)
#
# RUNTIME 5-15 min. Qwen did 666 rows in 43m => ~1.4 min for 22 rows; gemma
# measured 1.5x Qwen on the tsar_cefr arm (16018827, 10m48 vs 7m11), and a
# 16 GB checkpoint loads in a few minutes. --time 1:00:00 is >=4x that.
#
# QUOTA: infrastructure (does not count, .claude/experiments.md item 5).
# ARTIFACTS: outputs/ergo_cap_smoke_gemma4/ is new; the Qwen arm's
# outputs/ergo_upB_random_excite/ is not touched.
# NO pip install (item 6): run_ergo_math_screening.py puts src/ on sys.path
# itself, so the editable-install race of 2026-09-08 is structurally impossible.
# It also uses transformers, not vLLM, so it cannot contend for the vLLM
# compile cache the way 15943452 did.

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
  --output-dir outputs/ergo_cap_smoke_gemma4 \
  --num-items 4 --item-rng-seed 2 \
  --seeds 0 \
  --controller random_excite --random-excite-p 0.5 \
  --reset-mode append \
  --prompt-profile upstream
