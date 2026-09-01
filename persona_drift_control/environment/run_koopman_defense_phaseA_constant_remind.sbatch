#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseA
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseA-%j.out

# Phase A of docs/experiments/koopman_defense_pilot.md: actuator-authority
# check for the channel-A-style safety-reminder actuator (safety_reminder.py)
# on the adversarial-defense screening plant. --controller constant_remind
# (safety reminder inserted every turn) on the SAME 20-attack panel and
# seeds as the non-thinking baseline (job 15399715, --attack-rng-seed 0,
# --num-attacks 20, --seeds 0 1) so the two runs' new-Q1 numbers are
# directly comparable without re-running the baseline (its data/report
# already exist under outputs/adversarial_screening/).
#
# --time is the same generous upper bound as run_adversarial_screening.sbatch
# (same trajectory count/turn count; the extra reminder tokens per turn are
# a small fraction of agent_max_new_tokens=256's context, not expected to
# meaningfully change wall-clock). Resumable (adversarial_screening.py) --
# a --time kill just needs `sbatch` on this same script again.

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
  --output-dir outputs/koopman_defense_phaseA_constant_remind \
  --num-attacks 20 --seeds 0 1 --attack-rng-seed 0 \
  --controller constant_remind
