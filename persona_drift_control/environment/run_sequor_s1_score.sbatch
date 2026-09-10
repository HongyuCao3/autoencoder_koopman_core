#!/bin/bash
#SBATCH --job-name pdc-sequor-s1-score
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 04:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s1-score-%j.out

# S1's TWO READOUTS (instrument) -- judging only, no generation.
# Submit AFTER run_sequor_s1_arm.sbatch has completed and its rows are on disk.
#
# WHAT IT DECIDES: it produces the numbers every S1 gate and the S2 fit read.
# Independent judge (Qwen3-14B, cap 512) is the ONLY source of reportable
# numbers; the self judge (the agent's own model, cap 1024) is the in-loop
# SELECTION signal S3 will use and is never reported (.claude/global.md ->
# reporting discipline). Split into its own job for the reason 15763577
# demonstrated: judging a second time is cheap, regenerating is not.
#
# RUNTIME ESTIMATE 1.6-2.4 h. Assumptions: 9600 rows x 3 constraints = 28,800
# calls per judge at a measured 0.11 s/call (pilot: 8640 calls in 15.4 min on
# the 14B, 15.7 on the 4B) = ~53 min each, plus ~4 min to load each model.
# UNMEASURED: nothing new -- both rates come from the same models at the same
# caps on the same task. --time 04:00:00 is 1.7x the pessimistic end.
#
# EXPECTED, NOT A FAILURE: the self judge loses ~4-5% of rows to a parse
# failure at cap 1024 (pilot 121/2880 = 4.2%, of which 41 truncated). That is
# the number S3's controller will be blind on, and it is reported, not fixed by
# raising the cap here -- the cap is what the calibration was run at.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

ARM=outputs/sequor_s1_arm

echo "=== phase 1/2: reporting-judge readout (the only one any number may come from) ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --out-path "$ARM/readout_independent_qwen3_14b.json"

echo "=== phase 2/2: in-loop self-judge readout (selection signal, never reported) ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --judge-kind self \
  --max-new-tokens 1024 \
  --out-path "$ARM/readout_self_qwen3_4b_instruct_2507.json"
