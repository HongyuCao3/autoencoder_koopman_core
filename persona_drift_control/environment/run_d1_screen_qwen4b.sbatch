#!/bin/bash
#SBATCH --job-name pdc-d1-qwen4b
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 08:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/d1-screen-qwen4b-%j.out

# D1 baseline-rate screening (docs/experiments/two_task_success_plan.md
# section four, D1; Opus's ruling on the exclusion set / stratified
# 100-attack sample / new thin CLI). 100 attacks (conf/experiment/
# d1_attack_ids.txt, built by scripts/select_d1_attacks.py) x 1 seed x up
# to 5 turns each = up to 500 rows, zero control, no defense controller,
# --agent-max-new-tokens 1024 (raised from the 256/512 default per the
# spec -- B1's len1024 finding: truncation matters even though it doesn't
# move the baseline rate). New output dir, does not overwrite anything.

set -euo pipefail
module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
nvidia-smi -L || true
python -m pip install -e . -q

python scripts/run_d1_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/d1_screen_qwen4b \
  --attack-ids-file conf/experiment/d1_attack_ids.txt \
  --seeds 0 \
  --agent-max-new-tokens 1024
