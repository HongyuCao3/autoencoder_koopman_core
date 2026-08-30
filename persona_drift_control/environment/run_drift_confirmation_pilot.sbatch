#!/bin/bash
#SBATCH --job-name pdc-drift-confirmation
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 08:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/drift-confirmation-%j.out

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

# Follow-up power-analysis pilot (see docs/experiments/drift_confirmation_pilot.md):
# 10 prompts x 2 seeds x 2 conditions = 40 trajectories, live user-sim mode
# (scripted mode tried and rejected -- see that doc's history section).
# screening.py's resume support (append + skip-completed-trajectory-id) means
# if this hits --time before finishing, resubmitting this SAME sbatch script
# continues from wherever trajectories.jsonl left off instead of restarting.
python scripts/run_signal_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --num-prompts 10 --seeds 0 1 --prompt-rng-seed 0 \
  --user-mode live \
  --output-dir outputs/drift_confirmation_pilot
