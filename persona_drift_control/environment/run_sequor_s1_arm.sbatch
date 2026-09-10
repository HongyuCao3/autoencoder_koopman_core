#!/bin/bash
#SBATCH --job-name pdc-sequor-s1-arm
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 08:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s1-arm-%j.out

# S1 IDENTIFICATION ARM (method) -- generation only, judging is a second job.
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md).
#
# WHAT IT DECIDES: it IS the mainline. These rows are simultaneously premise
# (i)'s test (S1a), the Koopman training set (S2 fits y_{t+1}=Ay_t+Bu_t+c on
# S1b), and S3's zero-control lower-bound arm. Plan section 4.
#
# SIZED FROM MEASUREMENT, NOT ASSUMPTION. The pilot (15768661) replaced the
# plan's assumed per-item difference sd of 0.30 with a measured one, split into
# the two components that enter the MDE differently
# (outputs/sequor_s1_pilot_sizing/s1_sizing_report.json, 11 items x 3 seeds,
# independent judge, tuple_294_1 excluded):
#
#   late window (t15..t20)  contrast +0.1397  CI [+0.0572, +0.2205]
#                           sd between items 0.1317, within item over seeds 0.1000
#                           -> MDE(n=40, s=3) 0.0636   effect/MDE 2.20
#   turn 20 only            contrast +0.1515  CI [+0.0303, +0.2828]
#                           sd between items 0.2054, within item over seeds 0.1535
#                           -> MDE(n=40, s=3) 0.0991   effect/MDE 1.53
#
# The between-item term dominates, so seeds past 3 buy almost nothing
# (MDE(n=40) 0.0636 at s=3 vs 0.0616 at s=5) and only N moves the MDE. Hence
# N=40, s=3: it is the plan's N, and it is now the N the arithmetic supports.
#
# THE ESTIMAND MUST BE SIGNED BEFORE THIS JOB IS SUBMITTED. Proposed: S1a is
# the per-item difference of mean `y` over the LATE WINDOW t15..t20,
# independent judge, paired within (item, seed), with the turn-20-only
# functional reported beside it as a secondary. This is the agent's proposal,
# NOT a user ruling -- the 2026-09-10 rulings covered the item set and the
# fidelity wording. The two windows imply MDEs a factor 1.6 apart, so whoever
# signs it is choosing the design's power; picking one after S1's rows exist
# would be choosing the estimand from the noise.
#
# ITEM SET: the first 40 dialogues by conversation_id from the whole bank
# (`--item-selection bank`), user ruling 2026-09-10 = option (a). What it
# costs, verbatim, wherever these rows produce a number: only 3 of the 40
# dialogues are fully gold-covered, and 78 of 120 judged constraints (65%) sit
# OUTSIDE the judge's calibration set. The judge's accuracy there was never
# measured; on the calibrated set it is Gold-Yes 0.8184 / Gold-No 0.9699.
# Judge error on the state is measurement noise on the regressor of K2/G-S2-1,
# which biases the state coefficient TOWARD ZERO -- the direction that would
# fake this line's death (ERGO died at 97% attenuation). The clean purchase of
# that cell is a small human check set on UNCOVERED constraints (screening
# section 9.2); it is zero-GPU and still outstanding.
#
# RESPONSE CAP: 2048, unchanged, and the per-item >5% criterion still governs.
# Job 15768372's probe named 6 dialogues that cannot run; NONE is in this set.
# But that probe FAILED ITS OWN POSITIVE CONTROL (it missed tuple_294_1, whose
# length grows with turn depth), so it gives a lower bound only. Three items in
# this set sit in the band that crossed at 20 turns: tuple_139_1 (3-turn median
# 1861), tuple_128_1 (1522), tuple_131_1 (1500). EXPECT 1-3 items to bind the
# cap. The standing ruling applies unchanged: do NOT raise the cap and do NOT
# drop items here -- the arm reports per-item cap accounting, and the only way
# past the analyzer's guard is a named --exclude-item recorded beside the
# numbers it changes (screening section 10 items 4 and 6).
#
# WHAT THE PILOT DOES NOT TRANSFER: its 12 items were the fully gold-covered
# ones and only 3 of them are in this set, so the between-item sd was measured
# on a judge-friendly, narrower mix. If it inflates 1.5x on this mix, MDE(n=40,
# s=3) becomes ~0.092 -- still below the pilot's +0.1397 contrast, which is the
# margin N=40 was chosen for. If the S1 contrast lands under its own MDE, the
# pre-registered consequence is to report and hand back, NOT to close the line:
# executor authority was already answered at one step by K3 (+0.1233, CI
# [+0.0595, +0.1898], 1.30x MDE).
#
# RUNTIME ESTIMATE 3.0-5.9 h. Assumptions: 4 arms x 40 items x 20 turns x 3
# seeds = 9600 generations, batch width 40 (one batch per arm-seed-turn).
# Measured rates, all at batch 12: pilot 2.07 s/generation with sustained
# re-injection, cap-2048 branch arm 2.01, fidelity-harness arm 2.23. Batch 40
# is 3.3x wider and the pilot was not saturating an 80GB card, so per-generation
# time should fall; the probe measured 0.23 s/generation at batch 200, but only
# over 3 turns, so it does not bound a 20-turn prefill. Taken as 1.1-2.2
# s/generation -> 2.9-5.9 h. UNMEASURED: throughput at batch 40 with 20-turn
# contexts, and whether `--gpus a100:1` lands a 40GB card (at max_model_len
# 49152 a 40GB card holds ~4 concurrent requests vs ~9 on 80GB, which is the
# pessimistic end). --time 08:00:00 is 1.35x the pessimistic end; the phase
# writes trajectories.jsonl before any summary, so a timeout loses the tail of
# the run, not the run (job 15761810's lesson, now covered by a unit test).
#
# NOT INCLUDED HERE: judging. 9600 rows x 3 constraints x 2 judges = 57,600
# calls at a measured 0.11 s = ~1.8 h, submitted as its own job
# (run_sequor_s1_score.sbatch) so that a judging failure cannot cost the
# generation, the way 15763577 recovered 15761810's rows.
#
# QUOTA: method, by the EK-A precedent already signed for this line. Last 10
# project jobs stand at instrument 4 / method 5 / infra 1; this makes it 4/6/1.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_sequor_s1_pilot_arm.py \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --item-selection bank \
  --n-items 40 \
  --n-turns 20 \
  --seeds 0 1 2 \
  --arms zero_control constant_remind bernoulli antithetic \
  --constraints-in-system \
  --decoding model_default \
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --mode canonical \
  --out-dir outputs/sequor_s1_arm
