#!/bin/bash
#SBATCH --job-name pdc-pressure-screening
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 04:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/pressure-screening-%j.out

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

# Persona-pressure confirmation pilot (see
# docs/experiments/drift_confirmation_pilot.md, "下一步的小范围实验建议"):
# 4 prompts (2 per category) x 2 seeds x 2 conditions (baseline/live vs
# escalating_pressure) = 16 trajectories, 16 turns each. Estimated from
# drift_confirmation_pilot.md's measured ~414s/trajectory average (40
# trajectories in 16585s): worst case (character_traits-heavy) ~2.3h, budget
# below leaves generous margin. run_pressure_screening resumes the same way
# screening.py does -- if this hits --time, resubmit this same sbatch script.
python scripts/run_pressure_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --num-prompts 4 --seeds 0 1 --prompt-rng-seed 0 \
  --output-dir outputs/pressure_screening
