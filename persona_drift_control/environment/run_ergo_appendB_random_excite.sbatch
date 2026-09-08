#!/bin/bash
#SBATCH --job-name pdc-ergoAB-excite
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 06:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoAB-excite-%j.out

# E3 Step 1 (docs/experiments/two_task_success_plan.md section 2 E3):
# Phase B' -- append-mode open-loop random-excitation data collection for
# the append-mode Koopman identification fit, same 60-item draw as Phase B
# (--item-rng-seed 2), --reset-mode append. Copied from
# environment/run_ergo_phaseB_random_excite.sbatch verbatim except
# job-name, log path, --reset-mode append, --output-dir, and --time
# (widened to 06:00:00: append-mode prompts grow monotonically, so this
# runs slower than overwrite's 01:00:00 template).

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
  --output-dir outputs/ergo_appendB_random_excite \
  --num-items 60 --item-rng-seed 2 \
  --seeds 0 1 \
  --controller random_excite --random-excite-p 0.5 \
  --reset-mode append
