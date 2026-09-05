#!/bin/bash
#SBATCH --job-name pdc-mc-sycophancy-screening
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/mc-sycophancy-screening-%j.out

# First real run of the MMLU-based sycophancy screening
# (mc_sycophancy_bank.py/mc_answer_judge.py), which replaces
# run_sycophancy_screening.sbatch's SYCON-Bench item pool -- see
# docs/experiments/sycophancy_screening_pilot.md's ground-truth audit and
# resources/PROVENANCE.md's mmlu_sycophancy_mc.jsonl entry for why.
#
# Validation-scale run (20 items x 2 seeds x 5 turns = 200 turns, same size
# as the original SYCON-Bench screening) before committing to the full
# ~60-item sample-expansion scale next-step (a) called for: the point of
# this run is to check the new pipeline actually works end-to-end on a real
# model (regex extraction hit rate, turn-1 accuracy, no crashes) before
# spending more GPU time on it.
#
# --time is longer than run_sycophancy_screening.sbatch's 1 hour: agent
# max-new-tokens is 512 here (vs 256), since the agent prompt asks for
# step-by-step reasoning before a final-answer line, and --judge-model is
# loaded as a second model even though the regex-first design means it will
# rarely be called (only when regex extraction fails to find a "Final
# answer: X" line).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_mc_sycophancy_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/mc_sycophancy_screening \
  --num-items 20 --seeds 0 1 --item-rng-seed 0
