#!/bin/bash
#SBATCH --job-name pdc-tsar-cefr-gpu2a
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 6:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/tsar-cefr-gpu2a-%j.out

# tsar_cefr GPU-2a: COUNTERFACTUAL BRANCH TREE, depth 4 (method).
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md item 1).
#
# WHAT IT DECIDES. D-2.5, the causal upper bound -- the one gate that three
# closed lines all died on, and the only one that does not pass through a
# fitted operator. GPU-1 was pre-registered to answer it with zero GPU. That
# assumption failed, for a reason specific to how the two lines excite:
#
#   `defense`   every fixed_t{k} arm is byte-identical to zero_control for
#               turns 1..k-1, so ANY policy replays off disk exactly.
#   `tsar_cefr` GPU-1 drew six actions independently per trajectory. Each item
#               got 3 of 4^6 = 4096 paths; 0 of 199 items saw all four actions
#               from the identical step-1 state; branch points collapse 451 /
#               106 / 28 / 10 / 1 across steps 1..5; the IPW effective sample
#               for ONE 6-step policy is 597/4096 = 0.146 trajectories.
#
# The estimator was never wrong -- uniform random actions make IPW unbiased.
# The POWER is wrong, and power here is bought with counterfactuals, not with
# N: doubling the trajectories still leaves 0.29 of a trajectory per candidate
# policy. This job buys the counterfactuals directly, so the horizon-4 bound is
# EXACT: no operator, no importance weight, no policy evaluated at a state it
# never visited.
#
# DEPTH 4 IS THE PLAN'S OWN HORIZON (section 5.4: the MPC optimises over H=4),
# not a budget compromise. Depth 6 would be 1.1M generations and is not on the
# table. Depth 3 (16,716 generations, ~35 min) is a STRICTLY WEAKER bound, so a
# depth-3 zero would already close the line -- that is the cheap fallback if
# this estimate proves optimistic, and it is sound in the direction that
# matters.
#
# WHAT ELSE THE TREE PAYS FOR, at no extra GPU. Every arm in plan section 5.4
# is a policy over these same four actions, so `one_shot`, `fixed_ladder`,
# `greedy_reactive` and `koopman_mpc` can all be READ OFF this tree exactly at
# H=4 instead of being generated again. `dp_path` cannot -- see the blocker
# below. This also gives the operator from Phase 3 a ground truth to be scored
# against: the MPC's predicted rollout versus what the model actually does.
#
# BLOCKER THAT MUST BE RULED ON BEFORE THIS BUYS A D-2.5 VERDICT. D-2.5 is
# "supremum - dp_path", and `dp_path` has NO definition available: no PDF, no
# bib entry, no code, no reward matrix, no DP state/action definition anywhere
# in the repo -- only the paper's NAME, in plan sections 3.1/5.4. Phase 1's
# provenance check (resources/tsar_cefr/README.md) covered the dataset, the
# three CEFR classifiers, MeaningBERT and the official scorer, and MISSED this
# one, though section 3.1 ranks it the first-priority instrument. The tree is
# worth generating either way (it is the supremum's only source), but the
# subtrahend needs a user ruling.
#
# ALSO AWAITING SIGNATURE, BEFORE THE TREE EXISTS. The POLICY CLASS. With a
# continuous `ell` every node carries a unique state, so an unrestricted
# "xi-measurable" supremum degenerates to the clairvoyant best path. The
# `defense` precedent this gate points at did not do that: it binned the
# readout (3 bins, 4 as secondary) and let the LEAVE-ONE-OUT number decide,
# with the in-sample figure reported beside it (+0.0469* in-sample on 16
# trajectories became +0.0042 crossing zero on 40). An upper bound whose policy
# class is chosen after the tree is on disk is not an upper bound.
#
# SCALE: 199 items (100 test sources x 2 targets, minus the named exclusion
# `51-a2`) x (4 + 16 + 64 + 256) = 340 nodes per item = 67,660 generations and
# 67,859 readout texts. Dry-run verified with the real argparse.
#
# RUNTIME ESTIMATE 1.4-2.9 h (--time 6:00:00 = ~2.1x the pessimistic end).
# ASSUMPTIONS, all rates MEASURED ON GPU-1 (15850523) rather than assumed --
# the LEDGER explicitly says not to reuse GPU-1's own pre-run estimate:
#   - GENERATION 0.02 s/gen: GPU-1 logged "600 generations in 0.2 min" at every
#     one of six steps, batch 600, A100 80GB. 67,660 x 0.02 = 23 min. Batches
#     here are larger (chunked at 8,192), which should not be slower; the band
#     allows 0.02-0.05 s/gen -> 23-56 min.
#   - READOUT 0.059 s/text: GPU-1's readout ran 21:16:51 -> 21:20:38 (3.8 min)
#     over ~3,880 texts through 3 ModernBERT-base classifiers + MeaningBERT at
#     batch 32. 67,859 x 0.059 = 67 min; band 60-110 min. READOUT, NOT
#     GENERATION, IS THE DOMINANT COST HERE -- a 19x row count against a 19x
#     cheaper per-row model.
#   - MODEL LOAD ~2-4 min (GPU-1: job start 21:13:27, engine init 21:14:15).
#   - PROMPT LENGTHS ARE UNCHANGED: this task is paragraph-level and context
#     does NOT accumulate (the LEDGER records this as why GPU-1's estimate was
#     8x conservative). The cap is 1.5x the SOURCE length, fixed for the whole
#     tree exactly as in GPU-1, so the two arms' rows stay comparable.
#   NOT MEASURED: vLLM 0.28 throughput at 8,192-prompt batches on this node,
#   and whether host memory holds 67,660 rows with both texts (~200 MB, well
#   inside 96G). The runner writes tree_raw.jsonl BEFORE the readout pass, so
#   an OOM in the classifiers costs the CPU-cheap half only.
#
# QUOTA (item 5): the last GPU job was 15850523 (method); no GPU job has run
# since. The window at that submission was instrument 1 / method 8 / infra 2,
# and GPU-1 entering it as METHOD can only lower the instrument count. This job
# is METHOD. Instrument axis keeps >= 4 slots of headroom.
#
# ARTIFACTS: outputs/tsar_cefr_gpu2a_branch/ does not exist (dry-run checked;
# the runner refuses to write into an existing directory). GPU-1's rows are
# read by nothing here and are not touched.
#
# NO pip install here (item 6): the runner reaches src/ by file path, and this
# is a single job with nothing else of this project queued.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_tsar_cefr_branch_arm.py \
  --agent-model Qwen/Qwen3-4B \
  --split test \
  --depth 4 \
  --temperature 6.5 \
  --readout-device 0 \
  --readout-batch-size 32 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --chunk 8192 \
  --mode canonical \
  --out-dir outputs/tsar_cefr_gpu2a_branch
