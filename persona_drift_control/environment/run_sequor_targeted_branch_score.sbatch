#!/bin/bash
#SBATCH --job-name pdc-sequor-targeted-score
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 02:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-targeted-score-%j.out

# READOUT FOR THE TARGETED-REMINDER PILOT (instrument).
# Submit with --dependency=afterok:<pilot job id>, AFTER a user ruling.
#
# WHAT IT DECIDES: where the pilot's numbers come from. The independent 14B is
# the only source any reported number may have (.claude/global.md, reporting
# rules). Judging is a separate job from generation for the reason job 15763577
# established: re-judging is cheap, re-generating is not.
#
# ONLY the independent judge runs here. The in-loop 4B is not needed: the
# targeting decision was made OFFLINE from the readouts that already exist, and
# what a 4B-driven controller would have targeted instead is already counted at
# zero GPU in the pilot's arm_report (`targeting_agreement`: identical on
# 48.6% of the 1104 points, different on 234, and it would not have acted at
# all on 75 unparsed + 258 it believes are clean).
#
# RUNTIME ESTIMATE 16-25 min (--time 02:00:00 = ~5x). ASSUMPTIONS: 2208 rows x
# 3 constraints = 6624 judge calls at the 0.11 s/call measured on 15772957
# (28800 calls), = 12 min, plus ~4 min load. Cap 512 is the 14B's measured
# setting (12/2000 truncations at S0, 2/9600 on the S1 arm).
#
# ARTIFACT: the readout is written into the pilot's own directory under a new
# name; the scorer refuses an existing --out-path.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

ARM=outputs/sequor_targeted_branch_pilot

python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --out-path "$ARM/readout_independent_qwen3_14b.json"
