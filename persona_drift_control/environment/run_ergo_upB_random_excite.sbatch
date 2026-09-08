#!/bin/bash
#SBATCH --job-name pdc-ergoUP-excite
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 06:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoUP-excite-%j.out

# R1.5 (docs/experiments/ergo_fidelity_restoration_plan.md section 16): the
# random-excitation identification arm under --prompt-profile upstream. b2 was
# last measured on the legacy-profile Phase B', which is now known to carry the
# same confound R1 removed, so this re-asks G-E3-1/G-E3-2 of a corrected
# harness with the thresholds unchanged. Copied from
# environment/run_ergo_appendB_random_excite.sbatch verbatim except job-name,
# log path, --prompt-profile upstream, --output-dir and --time.
#
# --time 06:00:00, not the source file's value: upstream-profile replies are
# much longer (G-R1-1 measured a final-turn median of 563 characters against
# legacy's 18) and R1's three arms took 37-44 minutes where the legacy append
# arms took 9-23. Estimated, not copied (failure mode 14).

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
  --output-dir outputs/ergo_upB_random_excite \
  --num-items 60 --item-rng-seed 2 \
  --seeds 0 1 \
  --controller random_excite --random-excite-p 0.5 \
  --reset-mode append \
  --prompt-profile upstream
