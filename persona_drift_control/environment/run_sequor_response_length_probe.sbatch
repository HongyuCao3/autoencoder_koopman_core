#!/bin/bash
#SBATCH --job-name pdc-sequor-len-probe
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 01:30:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-len-probe-%j.out

# BANK-WIDE RESPONSE-LENGTH PROBE (infrastructure -- not in the quota).
# User ruling 2026-09-09 evening ("能尝试的都提交").
#
# WHAT IT DECIDES: S1's response cap, and which of the 200 dialogues cannot be
# run under it. S0-0 ran on 12 hand-inspectable items; S1 wants N=40 drawn from
# the whole bank, and the per-item cap criterion has already fired twice, both
# times because of the constraint text rather than the model -- job 15756689 at
# 18.6% overall with `tuple_294_1` at 100%, and the fidelity-harness arm at
# cap 2048 with that same item at 21.4% while the other eleven sat at 0.0%.
#
# WHY NOT A REGEX: tried first, and it is sensitive but not specific. It flags
# both observed long items, and also flags `tuple_245_1`, whose median is 1294
# tokens and which never hit the cap. Sizing an arm on it would exclude items
# that are fine and would still miss any long item phrased differently.
#
# WHY 3 TURNS SIZE A 20-TURN ARM: response length here is a function of the
# constraint set, not of context depth. Job 15756689's truncations were flat
# across turns (~4 of 24 at every turn, no drift as context grew) while two of
# twelve items carried 75 of 87. That is the assumption, and it is stated in
# the script's docstring as an assumption: an item near the boundary at turn 3
# can cross it by turn 20, so this sizes the cap and names items, it does not
# certify an arm.
#
# HARNESS: the fidelity one (constraints in a system message + the model's own
# sampling defaults), because that is what S1 will use and it is what grew
# tuple_294_1's median from 1653 to 1858 tokens.
#
# RUNTIME ESTIMATE 12-35 min. Assumptions, all from measured runs: 200 items x
# 3 turns x 1 seed = 600 generations at cap 2048. The cap-2048 S0-0 arm ran
# 2.01 s/generation at batch 12 and the fidelity arm's sampled variant 2.10;
# batch width here is 200, but `--max-model-len 12288` puts real concurrency
# around 35-40 sequences on an 80GB card (the S0-0 arm measured 62.6 GiB of KV
# cache / 456k tokens at 49152), so roughly 3x the batch-12 throughput =>
# 0.7-2.1 s/generation => 7-21 min, plus ~4 min of model load. A 40GB card
# halves the concurrency; --time 01:30:00 is >= 2.5x the pessimistic end.
#
# NOT EVIDENCE. No K gate reads this artifact and no number from it goes in the
# paper. It decides a generation setting and an item list.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Job 15738603 died in FlashInfer's JIT sampler for want of nvcc on the compute
# node. This probe samples, so it exercises the sampler for real.
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

# No --n-items: the whole 200-dialogue bank, so whichever N=40 rule is signed
# tomorrow, the measurement is already there.
python scripts/probe_sequor_response_length.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --n-turns 3 \
  --seed 0 \
  --constraints-in-system \
  --decoding model_default \
  --max-new-tokens 2048 \
  --max-model-len 12288 \
  --mode canonical \
  --out-dir outputs/sequor_response_length_probe
