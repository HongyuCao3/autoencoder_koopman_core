#!/bin/bash
#SBATCH --job-name pdc-ergoUP-fixed_last
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 06:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoUP-fixed_last-%j.out

# R1 (docs/experiments/ergo_fidelity_restoration_plan.md section 3.2): the single-reset-at-the-final-turn (fixed_last)
# arm under --prompt-profile upstream, i.e. the answer-format instruction
# delivered once in a system message instead of concatenated onto every user
# turn. Same held-out 58 items and --reset-mode append as E2's arms; copied
# from environment/run_ergo_appendC_fixed_last.sbatch verbatim except job-name,
# log path, --prompt-profile upstream, and --output-dir.
#
# Upstream condition this arm corresponds to: RECAP.
#
# zero_control is re-run rather than reused from the legacy profile: reusing
# it would compare two prompt structures against each other (failure mode 22).
#
# --time is 06:00:00 rather than the template's 01:00:00 for the same reason
# E2's arms were widened, plus upstream-profile replies are expected to be
# LONGER than legacy's (that is what G-R1-1 measures), so generation is
# slower still. Not copied from a template blindly (failure mode 14).

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
  --agent-model Qwen/Qwen3-4B --device cuda \
  --output-dir outputs/ergo_upC_fixed_last \
  --controller fixed_last \
  --reset-mode append \
  --prompt-profile upstream \
  --item-ids $ITEM_IDS \
  --seeds 0 1
