#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseE-zero-len1024
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseE-zero-len1024-%j.out

# B1 diagnostic step (docs/experiments/defense_line_redesign_plan.md section 4.1):
# does lifting the reply length cap from 256 to 1024 tokens raise the blind
# assisted-rate baseline above the ~10% D1 measured at 256 tokens? Only the
# zero_control arm -- this step does not compare defenses, only measures the
# baseline. Same 8 held-out attacks x seeds 0 1 as the original phaseE arms.

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
  --output-dir outputs/koopman_defense_phaseE_zero_control_len1024 \
  --seeds 0 1 \
  --controller zero_control \
  --agent-max-new-tokens 1024 \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
