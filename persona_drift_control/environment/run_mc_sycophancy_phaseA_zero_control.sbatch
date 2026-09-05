#!/bin/bash
#SBATCH --job-name pdc-mc-phaseA-zero
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/mc-phaseA-zero-%j.out

# Executor-authority check (Phase A analog) on the MMLU sycophancy item pool
# -- docs/experiments/mc_sycophancy_screening_pilot.md's next-step 2. Redoes
# the check docs/experiments/sycophancy_screening_pilot.md's "Phase A"
# section ran on SYCON-Bench (job 15567174/15567175, inconclusive at
# p=0.096/n=10) on ground-truth-verified data instead, so a null result here
# can't be blamed on a confound in `correct_letter` the way it partly could
# there. --controller zero_control half of the pair -- see the sibling
# *_constant_remind.sbatch. 30 items x 2 seeds, a fresh --item-rng-seed 1
# (independent item draw from the 60-item sample-expansion run, job
# 15568231). --time mirrors the 20-item validation run's actual 38 minutes,
# scaled up ~1.5x for 30 items plus reminder-token overhead.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_mc_sycophancy_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/mc_sycophancy_phaseA_zero_control \
  --num-items 30 --seeds 0 1 --item-rng-seed 1 \
  --controller zero_control
