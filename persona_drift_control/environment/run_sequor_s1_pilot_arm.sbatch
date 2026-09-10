#!/bin/bash
#SBATCH --job-name pdc-sequor-s1-pilot
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 06:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s1-pilot-%j.out

# S1 SIZING PILOT (method).
# User ruling 2026-09-09 evening ("能尝试的都提交").
#
# WHAT IT DECIDES: S1's N, and whether S1a can be sized at all.
#
# Plan section 4 sizes S1a on an ASSUMED per-item difference sd of 0.30 (MDE
# ~0.13 at n=40) and says in the same breath to recompute it from a smoke run
# before submitting. That recomputation has never been possible: nothing in
# this line has run a SUSTAINED arm. The S0-0 branch arm measures a one-step
# gain from a byte-identical prefix; S1a compares the ENDPOINT of a trajectory
# reminded every turn against one that never was. Different quantities,
# different variances -- and the ERGO line already paid for treating one MDE
# as if it belonged to the other (an endpoint MDE of 0.49 used to excuse
# one-step gates running at se 0.013).
#
# THE DESIGN, at pilot scale, on the SAME 12 items S0-0 used:
#   zero_control     u=0 everywhere              ] S1a dose contrast,
#   constant_remind  u=1 from turn 2             ] paired by item
#   bernoulli        u_t ~ Bern(0.5) from turn 2 ] S1b antithetic pairing,
#   antithetic       the exact complement        ] one reminded side per cell
#
# The item set is deliberately unchanged. N=40 needs an item-selection ruling
# (only 16 of 200 dialogues are fully gold-covered, screening section 9.2) that
# has not been made, and a pilot must not make it by default. Job 15768372 is
# measuring the response-length half of that ruling in parallel.
#
# WHAT IT BUYS: the per-item endpoint difference sd, hence S1a's real MDE at
# n=40; whether `y` still has range at turn 20 under a SUSTAINED reminder,
# which the branch arm cannot answer because it never walks a reminded
# trajectory; G-S1's u-balance and cell coverage on real rows; and throughput
# for S1's runtime estimate.
#
# NOT S1, AND NOT A VERDICT. n=12 against a plan written for n=40. No K gate
# reads it. Its endpoint numbers are a variance estimate, and the
# pre-registered consequence of a small pilot effect is to RE-SIZE S1 -- never
# to declare the executor's authority absent. K3 already answered authority at
# one step (+0.1233, CI [+0.0595, +0.1898], 1.30x MDE).
#
# RUNTIME ESTIMATE 2.2-3.1 h. Assumptions, all measured: 4 arms x 12 items x
# 20 turns x 3 seeds = 2880 generations at batch 12, the same width as the
# arms that produced the rates -- the cap-2048 arm 2.01 s/generation, the
# fidelity-harness arm 1404 generations in 52.2 min = 2.23. Sustained
# reminding re-injects the block every turn so prefill grows over a trajectory
# in a way neither of those measured; at 2.1-2.6 s/generation that is 100-125
# min, and 150 min pessimistically. Phase 2 is 2880 x 3 = 8640 judge calls at
# a measured 0.10 s = ~15 min plus 4 min to load the 14B; phase 3 the same on
# the 4B at cap 1024. --time 06:00:00 is ~2x the pessimistic end.
#
# QUOTA: method, by the EK-A precedent (15708895) already signed for this line
# -- these rows are identification data, not an instrument check. The last 10
# jobs stand at 5 instruments, which is the stop threshold, so an instrument
# could not be submitted tonight anyway.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Job 15738603 died in FlashInfer's JIT sampler for want of nvcc on the compute
# node. This arm samples, so it exercises the sampler for real.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

ARM=outputs/sequor_s1_pilot_arm

echo "=== phase 1/3: four schedule arms, 3 seeds, fidelity harness ==="
python scripts/run_sequor_s1_pilot_arm.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --n-items 12 \
  --n-turns 20 \
  --seeds 0 1 2 \
  --arms zero_control constant_remind bernoulli antithetic \
  --constraints-in-system \
  --decoding model_default \
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --mode canonical \
  --out-dir "$ARM"

echo "=== phase 2/3: reporting-judge readout (the only one any number may come from) ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-14B \
  --judge-kind independent \
  --max-new-tokens 512 \
  --out-path "$ARM/readout_independent_qwen3_14b.json"

echo "=== phase 3/3: in-loop self-judge readout (selection signal, never reported) ==="
python scripts/score_sequor_trajectories.py \
  --arm-dir "$ARM" \
  --judge-model Qwen/Qwen3-4B-Instruct-2507 \
  --judge-kind self \
  --max-new-tokens 1024 \
  --out-path "$ARM/readout_self_qwen3_4b_instruct_2507.json"
