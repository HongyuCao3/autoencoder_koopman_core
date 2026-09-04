#!/bin/bash
#SBATCH --job-name pdc-phaseJ-rejudge
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseJ-rejudge-%j.out

# Independent-judge re-scoring of the defense line's already-collected
# trajectories (Phase J's 7 arms + the 3 unbudgeted references
# analyze_budget_arm_comparison.py reports alongside them).
#
# Every judge-scored number on this line was produced with judge == agent ==
# Qwen/Qwen3-4B (conf/adversarial_screening.yaml's `judge_model: null`
# default). The sycophancy line's paired rerun (job 15487325) showed what
# that costs there -- one-way misses, ~4x compression of the effect size, a
# flattened inertia structure -- and this line has never checked. Same judge
# checkpoint as that rerun (Qwen/Qwen3-4B-Instruct-2507: same family,
# generation and size, genuinely different weights) so the two lines' bias
# numbers are directly comparable.
#
# No agent generation at all: the judge prompt is rebuilt from each stored
# row (plain_query / attacker_query / agent_message) and the judge seed is
# recomputed by rejudge.py from the row's own seed and turn, so this is one
# 10-token greedy judge call per row -- ~1500 rows today (the 5-seed
# expansion of fixed_t3/t4/t5 is still running; rerunning this script later
# picks up only the rows that appeared since, see rejudge.rejudge_file's
# resumability). One 4B model loaded, nothing else.
#
# --time 01:00:00: the original screening spent ~31 min on 200 rows of
# *agent generation plus judging*; judging alone at 10 greedy tokens is a
# small fraction of that per row, so an hour over ~1500 rows is a loose
# upper bound rather than an estimate. Resumable either way.
#
# Reads outputs/*/trajectories.jsonl, writes only
# outputs/*/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl -- the
# original self-judged files are never touched.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/rejudge_safety_runs.py \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda

# CPU-only, seconds -- run here too so the job's log already contains the
# comparison instead of needing a second manual step.
python scripts/compare_safety_judge_runs.py
