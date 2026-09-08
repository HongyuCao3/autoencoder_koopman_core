#!/bin/bash
#SBATCH --job-name pdc-ergoALT-fixed_last
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 06:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoALT-fixed_last-%j.out

# R4 (docs/experiments/ergo_fidelity_restoration_plan.md section 6 / 15.2):
# model control for G-R1-3's failed ordering. Upstream's model set is Phi-4 /
# Llama-3.1-8B / GPT-4o with no Qwen, and they report RECAP > SNOWBALL while
# R1 on Qwen3-4B gives SNOWBALL > RECAP. This re-runs R1's three arms on
# Qwen3-4B-Instruct-2507 (already in /scratch/hcao2/hf_cache/hub/, used by D1).
#
# RECORD ONLY -- G-R4-1 reports whether the ordering and the reply-length
# distribution match R1's. Do NOT run a cross-model paired comparison on these
# outputs against the Qwen3-4B arms (failure mode 6). This does not enter any
# headline result.
#
# Identical to environment/run_ergo_upC_fixed_last.sbatch except job-name, log path,
# --agent-model and --output-dir.

set -euo pipefail
module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
nvidia-smi -L || true
python -m pip install -e . -q

ITEM_IDS=$(cat conf/experiment/ergo_phaseC_item_ids.txt)
python scripts/run_ergo_math_screening.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 --device cuda \
  --output-dir outputs/ergo_upC_alt_fixed_last \
  --controller fixed_last \
  --reset-mode append \
  --prompt-profile upstream \
  --item-ids $ITEM_IDS \
  --seeds 0 1
