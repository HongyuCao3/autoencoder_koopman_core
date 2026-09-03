#!/bin/bash
#SBATCH --job-name pdc-sycophancy-indep-judge
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sycophancy-indep-judge-%j.out

# Harness-validation rerun of run_sycophancy_screening.sbatch (job 15483493,
# 2026-09-02): SAME agent model (Qwen/Qwen3-4B, this project's default
# throughout), SAME 20 items / seeds 0 1 / item-rng-seed 0 -- the only
# variable changed is the judge, from self-judging (agent==judge, the
# default) to an independent same-generation, same-size checkpoint
# (Qwen/Qwen3-4B-Instruct-2507, a genuinely different set of weights --
# July-2025 non-thinking-only refined release vs. the original April-2025
# Qwen3-4B hybrid thinking/non-thinking model). Isolates the self-judging-
# bias hypothesis raised in docs/experiments/sycophancy_screening_pilot.md
# without also changing model family/generation/size, unlike the earlier
# Qwen2.5-7B-Instruct proposal would have. Separate --output-dir so both
# runs' trajectories/reports stay on disk for direct comparison.
#
# Two 4B-class models loaded at once (~8GB bf16 each) -- trivial on any A100
# tier this cluster hands out (checked historical slurm_logs: a mix of
# A100-SXM4-40GB and A100 80GB PCIe). --time budget unchanged from the
# original run: per-turn compute is the same (1 agent generate + 1 judge
# classify per turn either way), only the one-time cost of loading a second
# model differs.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_sycophancy_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --output-dir outputs/sycophancy_screening_independent_judge
