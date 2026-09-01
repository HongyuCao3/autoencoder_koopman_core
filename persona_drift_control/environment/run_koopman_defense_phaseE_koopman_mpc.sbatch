#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseE-mpc
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseE-mpc-%j.out

# Phase E, koopman_mpc arm -- the actual method being validated. Loads
# outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json's
# "richer_abs_sign" fit (nu=1, mu=2, chosen over the originally-planned
# nu=1/mu=1 and over mu=3 -- see docs/experiments/koopman_defense_pilot.md
# Phase C section), horizon=2, no repeat penalty. See
# run_koopman_defense_phaseE_zero_control.sbatch for the shared held-out
# attack panel this and the other 3 arms all use.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseE_koopman_mpc \
  --seeds 0 1 \
  --controller koopman_mpc \
  --koopman-model-path outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json \
  --koopman-model-key richer_abs_sign --koopman-nu 1 --koopman-mu 2 \
  --koopman-horizon 2 --koopman-repeat-penalty 0.0 \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
