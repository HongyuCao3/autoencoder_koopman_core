#!/bin/bash
#SBATCH --job-name pdc-mc-phaseA-t2
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/mc-phaseA-t2-%j.out

# Executor-authority check, reminder-design variant B: docs/experiments/
# mc_sycophancy_screening_pilot.md's Phase A section found constant_remind
# (reminder on every turn, including turn 1) a clean null against
# zero_control (job 15570373/15570374, paired t-test p=0.704 at n=22
# turn-1-baseline-gated items). This arm tests whether that null is
# specific to the "insert before turn 1 too" design: --controller
# fixed_schedule --fixed-schedule-turns 2 3 4 5 reminds on every pushback
# turn but never turn 1, since the reminder text presupposes pushback has
# already happened. SAME 30 items as job 15570373/15570374 (--item-ids,
# not --item-rng-seed) so this is directly paired against the existing
# zero_control run without needing to rerun it.

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
  --output-dir outputs/mc_sycophancy_phaseA_remind_from_turn2 \
  --seeds 0 1 \
  --item-ids mmlu_mc_0026 mmlu_mc_0106 mmlu_mc_0117 mmlu_mc_0132 mmlu_mc_0176 mmlu_mc_0213 \
    mmlu_mc_0220 mmlu_mc_0244 mmlu_mc_0260 mmlu_mc_0265 mmlu_mc_0328 mmlu_mc_0340 mmlu_mc_0422 \
    mmlu_mc_0431 mmlu_mc_0436 mmlu_mc_0437 mmlu_mc_0461 mmlu_mc_0570 mmlu_mc_0628 mmlu_mc_0665 \
    mmlu_mc_0684 mmlu_mc_0687 mmlu_mc_0721 mmlu_mc_0777 mmlu_mc_0895 mmlu_mc_0898 mmlu_mc_0910 \
    mmlu_mc_0914 mmlu_mc_0947 mmlu_mc_0998 \
  --controller fixed_schedule --fixed-schedule-turns 2 3 4 5
