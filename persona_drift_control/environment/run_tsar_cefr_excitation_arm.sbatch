#!/bin/bash
#SBATCH --job-name pdc-tsar-cefr-gpu1
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 3:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/tsar-cefr-gpu1-%j.out

# tsar_cefr GPU-1: OPEN-LOOP RANDOM EXCITATION (method: identification data).
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md item 1).
#
# WHAT IT DECIDES. Four of the seven death conditions at once, which is why
# this is one job and not four (kill_criterion.md section 2):
#   D-0    saturation -- does Qwen3-4B land on the target CEFR level in ONE
#          move? Above 80% there is no regulation problem left to solve.
#   D-1(2) post-run range -- does `ell_t` still take >= 3 distinct values at
#          EVERY step, on real trajectories rather than on 40 trial pairs?
#   D-2    executor authority -- does `step_down` beat `copy` on the next step,
#          CI excluding 0 and effect >= this arm's own MDE?
#   D-2.5  the causal upper bound: the in-sample supremum over all policies
#          measurable in `xi_t`, minus `dp_path`, computed on THESE rows BEFORE
#          any operator is fitted. This is the one that decides the line.
# D-2 and D-2.5 are computed by a zero-GPU analyzer on the rows this job
# writes; the job itself reports only admission, D-0 and D-1(2).
#
# WHY IT IS ALLOWED TO RUN AT ALL. G-T1 PASSED on 2026-09-14 -- but under a
# criterion 1b that was REVISED AFTER THE DATA WAS SEEN, from "100% of 40 trial
# pairs ordered correctly" to "direction correct on >= 85% and the paired mean
# Delta-ell CI upper bound below 0". The original threshold demanded zero error
# from an evaluator whose published precision is weighted-F1 0.89 / RMSE 0.34;
# at a 2% per-pair inversion rate a clean 40-pair sweep happens only 45% of the
# time. The four-point record, including "this is a post-hoc loosening", is in
# docs/experiments/tsar_cefr_results.md. Measured: 0.95 direction correct, mean
# Delta-ell -0.8415 CI [-1.0098, -0.6751] (bootstrap over the 20 source texts).
# EVERY number downstream of this job inherits that caption obligation.
#
# THE ONE INSTRUMENT REPAIR IS SPENT (temperature softening, T=6.5, 2026-09-13).
# If D-1(2) fires here, the line CLOSES -- there is no second repair.
#
# DESIGN, three points that look like mistakes and are not:
#   1. Decoding is GREEDY and the three seeds still differ. The seed draws the
#      ACTION SEQUENCE, which is the excitation this arm exists to create;
#      sampling noise on top would only inflate every gain's variance. 1 of 600
#      trajectories draws a duplicate sequence across seeds (dry-run measured,
#      expected at 4^6) and is counted in the report, not deduplicated.
#   2. `copy` is GENERATED, not spliced. It is the zero-input control that
#      identifies A, and what it must measure is what this harness does when
#      told to do nothing -- not what `cp` does. `unchanged_verbatim` reports
#      how often the model actually returned its input.
#   3. The readout runs AFTER generation, in a second pass. It can, because the
#      actions do not depend on the state -- that is what open-loop means -- so
#      the CEFR classifiers and vLLM never sit on the GPU together.
#
# THE MODEL IS Qwen/Qwen3-4B AND THAT IS A JUDGEMENT CALL AWAITING THE RULING.
# LEDGER section 5 signs "Qwen3-4B, same as the three behaviour lines,
# conf/task/defense.yaml", and that file says `Qwen/Qwen3-4B`. The `constraint`
# line's recent arms (S1, S3) run `Qwen/Qwen3-4B-Instruct-2507` instead. The
# repo splits 119 references to the former against 49 to the latter. This job
# follows the line's own written source; `enable_thinking=False` is set either
# way, which is what makes Qwen3-4B usable in a rewriting harness at all (a
# reasoning preamble would land in `text_out` and become step t+1's INPUT).
# IF THE RULING PREFERS Instruct-2507, change one flag -- nothing else moves.
#
# SCALE: 100 test sources x 2 target levels x 3 seeds x 6 steps = 3,600
# generations, + 40 saturation probes on the trial split = 3,640.
#
# RUNTIME ESTIMATE 0.4-1.0 h (--time 3:00:00 = ~3x the pessimistic end).
# ASSUMPTIONS, each with its source:
#   - Source paragraphs measured on the pinned test split with this model's
#     tokenizer: 102 tokens at p50, 146 at p90, 175 max. The per-item cap is
#     1.5x the SOURCE length (plan section 4.1), fixed for all six steps so it
#     cannot drift as the text gets simpler: 154 at p50, 263 max, 97k tokens of
#     headroom per sweep across 600 trajectories.
#   - <= 583k output tokens over six fully-batched sweeps, realistically ~400k
#     at 60-80% of cap. At 1,200-2,500 output tok/s for a 4B model on one A100
#     with 600-prompt batches -> 3-8 min of generation.
#   - + ~4 min model load (the S3 arm's measured load, same model class).
#   - + 5-15 min readout: 3,880 texts x 3 ModernBERT-base classifiers and 3,680
#     MeaningBERT pairs at batch 32, all short single paragraphs.
#   NOT MEASURED: vLLM 0.28 throughput on this node at these lengths, and
#   whether vLLM hands its KV cache back in-process. The runner therefore writes
#   trajectories_raw.jsonl BEFORE the readout pass -- if the classifiers OOM,
#   the 3,640 generations survive and only the CPU-cheap half is redone
#   (--readout-device -1 runs it on CPU).
#   Context is one paragraph, NOT an accumulating dialogue: --max-model-len 4096
#   is ~8x the longest prompt+completion and buys larger KV batches.
#
# QUOTA (.claude/experiments.md item 5): last 10 jobs are instrument 1 /
# method 8 / infra 2 (LEDGER section 5, unchanged since -- G-T1 was zero GPU).
# This job is METHOD. Instrument axis has 4 slots of headroom.
#
# ARTIFACTS: outputs/tsar_cefr_gpu1/ does not exist (dry-run checked; the runner
# refuses to write into an existing directory). Nothing existing is overwritten.
#
# NO pip install here (item 6): the runner reaches src/ by file path.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_tsar_cefr_excitation_arm.py \
  --agent-model Qwen/Qwen3-4B \
  --split test \
  --probe-split trial \
  --n-steps 6 \
  --seeds 0 1 2 \
  --temperature 6.5 \
  --readout-device 0 \
  --readout-batch-size 32 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --mode canonical \
  --out-dir outputs/tsar_cefr_gpu1
