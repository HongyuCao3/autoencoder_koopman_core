#!/bin/bash
#SBATCH --job-name pdc-d1-alt
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 08:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/d1-screen-alt-%j.out

# D1 baseline-rate screening, alternate target model
# (docs/experiments/two_task_success_plan.md section four, D1, Q3 amended
# per user approval 2026-09-07: Qwen/Qwen3-4B-Instruct-2507 instead of
# Llama-3.1-8B-Instruct, which is gated and not cached locally). Same 100
# attacks (conf/experiment/d1_attack_ids.txt) as run_d1_screen_qwen4b.sbatch
# x 1 seed x up to 5 turns each, zero control, no defense controller,
# --agent-max-new-tokens 1024. New output dir, does not overwrite anything.
# Do NOT run a cross-model paired comparison on these two arms' output --
# D1 only measures each arm's own baseline rate (failure mode 6).

set -euo pipefail
module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
nvidia-smi -L || true
python -m pip install -e . -q

python scripts/run_d1_screening.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/d1_screen_alt_qwen3_4b_instruct_2507 \
  --attack-ids-file conf/experiment/d1_attack_ids.txt \
  --seeds 0 \
  --agent-max-new-tokens 1024
