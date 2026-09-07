#!/bin/bash
#SBATCH --job-name pdc-soft-safety-judge
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/soft-safety-judge-%j.out

# F4 Step 1 of docs/experiments/signal_resolution_plan.md: score Phase B's
# 300 already-judged rows with the expected-value ("soft") safety judge
# instead of the greedy argmax digit -- same judge model, same prompt, one
# forward pass per row, no generation. Modeled on
# environment/run_phaseB_rejudge.sbatch's resource footprint.
#
# Only writes a new file (outputs/koopman_case_study/soft_judge_phaseB.json);
# never touches trajectories.jsonl.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/score_soft_safety_judge.py \
  --rows-path outputs/koopman_defense_phaseB_random_excite/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --out-path outputs/koopman_case_study/soft_judge_phaseB.json
