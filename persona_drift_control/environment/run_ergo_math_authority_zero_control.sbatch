#!/bin/bash
#SBATCH --job-name pdc-ergo-authority-zero
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergo-authority-zero-%j.out

# Minimal executor-authority check for the ERGO/Laban multi-turn-reliability
# candidate line -- docs/feasibility/ERGO_MULTITURN_RELIABILITY_FEASIBILITY.md
# section 4 step 2, run BEFORE any further bank/judge/trajectory engineering
# (the ordering inversion that section's whole point is: sycophancy's two
# reminder-design nulls only got tested after the full pipeline was already
# built). --controller zero_control half of the pair -- see the sibling
# *_constant_remind.sbatch, which uses the SAME 20 items (--item-ids, not
# --item-rng-seed, for a directly comparable paired run) but reset-every-
# turn. 20 items x 2 seeds = 222 total (item, turn) rows -- no separate
# judge model, no judge model calls at all (ergo_math_judge.py is pure
# regex), so this should be cheaper per-row than the MC sycophancy pilots
# despite the larger raw row count; --time is a generous upper bound, not a
# measured estimate (first run of this pipeline).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_ergo_math_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/ergo_math_authority_zero_control \
  --seeds 0 1 \
  --item-ids ergo_GSM8K_1247 ergo_GSM8K_825 ergo_GSM8K_1287 ergo_GSM8K_14 ergo_GSM8K_1240 \
    ergo_GSM8K_942 ergo_GSM8K_808 ergo_GSM8K_646 ergo_GSM8K_855 ergo_GSM8K_1260 ergo_GSM8K_1132 \
    ergo_GSM8K_107 ergo_GSM8K_417 ergo_GSM8K_1180 ergo_GSM8K_562 ergo_GSM8K_40 ergo_GSM8K_178 \
    ergo_GSM8K_1269 ergo_GSM8K_740 ergo_GSM8K_117 \
  --controller zero_control
