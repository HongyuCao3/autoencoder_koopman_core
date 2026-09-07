#!/bin/bash
#SBATCH --job-name pdc-ergo-entropy-readout
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergo-entropy-readout-%j.out

# NOTE (2026-09-07): the E0 verdict cited below was WITHDRAWN -- see
# docs/experiments/signal_resolution_plan.md sections 0.1/0.2 and task F0.
# E1 Step 1 (docs/experiments/backup/readout_controllability_gate_plan.md section
# 5.1): the ERGO line's E0 gate (scripts/analyze_ergo_readout_state.py)
# found y_task_success fails RC-1 and RC-3 -- binary, sparse (13.7% success),
# and almost entirely explained by the deterministic shard_frac ramp, not
# feedback-usable state. This job computes a token-entropy readout instead
# (teacher-forcing forward pass over each of the 666 already-recorded
# Phase B rows' agent_message), which section 5.2's CPU diagnostic (the
# same RC-2/RC-3 tests E0 ran) will then judge on its own.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/analyze_ergo_entropy_readout.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --rows-path outputs/ergo_math_phaseB_random_excite/trajectories.jsonl \
  --out-path outputs/ergo_math_phaseB_random_excite/entropy_readout.json
