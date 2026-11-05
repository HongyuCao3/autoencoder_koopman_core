#!/bin/bash
#SBATCH --job-name pdc-core-branch-sentlen
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 1:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/slurm/logs/core-branch-sentlen-%j.out

# #4b v2 SAME-PREFIX FORK ARM -- NOT SUBMITTED; WAITING FOR THE USER'S RULING.
# Spec: docs/experiments/command_channel_identifiability_plan.md section 3.5.
#
# WHY. v1 (19443006, slurm/run_core_excite_sentence_length.sbatch) randomised
# the ask on every turn, which broke "command = r - y_t", but its CB is still
# compared ACROSS prompts. This arm compares asks within one prompt and one
# history. Each trajectory runs the landed error protocol (trajectories.jsonl is
# the error corpus plus four command_* columns). Once per trajectory, at
# k ~ U{2,4,6} drawn from (seed, trajectory index), it forks: replies 1..k-1 and
# their earlier feedback are kept; the feedback after reply k-1 is replaced by
# the 4b template (truthful about the previous ask r, then asks c_j ~ U{5..100},
# j = 1..3, independent of target and readings). Each sibling generates reply k
# and stops. Siblings differ only in the number asked; written to
# branch_points.jsonl with k, prefix state (y_(k-1), y_(k-2)), asks, replies,
# git_sha, command_policy=branch, disable_thinking=true.
#
# ANALYSIS (fixed before collection): scripts/core_branch_b_identifiability.py
# -- within-fork paired CB, trajectory bootstrap 2000; _identifiability on all
# transitions (rank full and R^2 < 0.99). Gate: section 3.5, three rows.
#
# --disable-thinking IS REQUIRED (the runner refuses branch without it): the
# 09-17 arm 16035914 ran with thinking on, hit the 256-token cap on 90.6% of
# rows and counted words inside <think>.
#
# SCALE 3 seeds x 252 trajectories x (10 turns + 1 fork x 3 siblings)
# = 3 x (2520 + 756) = 9,828 generations.
#
# RUNTIME 8-20 min. ASSUMPTIONS: v1 19443006 ran 3 x 2520 generations of this
# task, runner, model (Qwen3-4B), decoding (greedy, max_new_tokens=256), thinking
# off, on one A100 in 6m11, including ~1.5 min engine start per seed; v2 is 1.3x
# the generations per seed. --time 1:00:00 leaves >3x headroom.
# NOTE: the main trajectories are greedy and use the error policy, so they are
# the same in all three seeds (only k and the sibling asks change with the seed,
# up to vLLM batch nondeterminism). That repeated work is ~4 min and is kept as
# specified.
#
# ENV: /scratch/hcao2/envs/sequor_vllm (rebuilt 2026-11-05; ran 19443006).
# PYTHONNOUSERSITE=1 IS REQUIRED: ~/.local/lib/python3.11 holds an old cu12
# stack whose nccl shadows torch's. /scratch can be purged again: verify
# `import vllm` in this env before submitting.
#
# QUOTA: method (identification data). Last 10 project jobs: instrument 1.
# ARTIFACTS: persona_drift_control/outputs/core_branch_sentence_length_<seed>/,
# none exists; the runner refuses an existing trajectories.jsonl or
# branch_points.jsonl. NO pip install. Uses vLLM: do not run beside another vLLM job.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core

nvidia-smi -L || true

for SEED in 0 1 2; do
  echo "=== sentence_length_t10 branch seed $SEED ==="
  python scripts/run_core_collection_arm.py \
    --task sentence_length_t10 \
    --agent-model Qwen/Qwen3-4B \
    --command-policy branch \
    --command-seed "$SEED" \
    --disable-thinking \
    --out-dir persona_drift_control/outputs/core_branch_sentence_length_$SEED
done
