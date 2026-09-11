#!/bin/bash
#SBATCH --job-name pdc-sequor-targeted-pilot
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 03:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-targeted-pilot-%j.out

# TARGETED-REMINDER PILOT (method, by the EK-A precedent: counterfactual pairs
# are identification data). Generation only; judging is a second job.
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md).
#
# WHAT IT DECIDES. S3's closed-loop arm has now died twice on paper, and both
# deaths share one mechanism: with a single on/off action the state can only
# SCALE what a reminder is worth, never move which turn is best to spend it on.
#   - 2026-09-10, scalar operator: 1 optimal schedule over 40 items, and over
#     101 starting states. MPC - best fixed = +0.0000024 vs MDE 0.0690.
#   - 2026-09-11, per-constraint binary state (ceiling represented explicitly):
#     1 schedule over the WHOLE state space. MPC - best fixed = +0.0003.
#   - 2026-09-11, threshold objective with priced reminders, matched cost:
#     largest gap +0.0084 vs that objective's own MDE 0.1142.
# This arm changes the ACTION instead of the model or the objective: restate
# only the constraint that broke. Which rule is broken at turn 12 is not
# knowable at turn 1, so a targeted reminder is something no fixed schedule can
# imitate. If it beats a blanket reminder, the closed loop has work to do and
# plan section 6's arm table is rewritten around a targeted actuator. If it
# does not, the action dimension is degenerate too and `koopman_mpc` is
# DROPPED rather than re-specified a fourth time.
#
# THE HINT IN THE DATA IT IS FOLLOWING (s2_koopman_fit_report_option_b.json):
# the three constraints of an item do not respond alike. Recovery gain of a
# blanket reminder on a constraint that is already broken --
#   slot 0  +0.0986 (n=155/160)   slot 1  +0.0108 (n=368/372)   slot 2  +0.0922 (n=325/339)
# a factor of 9 between them, which the scalar `y` averages away.
#
# PRE-REGISTERED, 2026-09-11, BEFORE SUBMISSION (screening section 10 item 13):
#   PRIMARY   recovery rate of a constraint broken at t-1: share judged
#             satisfied at t, TARGETED minus BLANKET, paired per
#             (item, seed, turn, constraint), bootstrapped over ITEMS.
#   MDE       0.0525. Not assumed: the item-level sd of this same recovery
#             contrast on the S1b arms is 0.1185 over 28 items with both cells
#             (1704 violated-constraint observations), and 2.8 x 0.1185 /
#             sqrt(40) = 0.0525. A blanket reminder buys +0.0589 on a broken
#             constraint, so this design resolves a targeted reminder that
#             roughly DOUBLES the actuator and nothing smaller. Below the MDE
#             is UNDECIDABLE, never "targeting does not work".
#   SECONDARY retention of the constraints ALREADY KEPT at t-1 (a targeted
#             reminder names them nowhere, and a model that reads the omission
#             as permission would trade one rule for another), and the net `y`
#             contrast. Reported WITH the primary, never instead of it.
#   COST      inserted_tokens per branch; the targeted block is shorter, so an
#             equal-token reading is available if the recovery rates tie.
#
# WHY IT IS ONLY ~1 GPU-HOUR: nothing is regenerated. The branches hang off the
# stored zero_control rows of outputs/sequor_s1_arm/, whose turn-t response was
# generated from exactly this prefix and IS the u=0 counterfactual. Every
# rebuilt prefix is hashed against the stored row's prefix_sha256 and the run
# aborts on a mismatch (tested). Both branches of a point share one sampling
# seed -- common random numbers, so the difference is not two draws.
#
# ONE ASYMMETRY, STATED RATHER THAN BURIED: the two fresh branches share a
# sampling seed with each other (common random numbers), but the stored u=0 row
# was drawn at a different time under sampled decoding, so it is NOT
# RNG-paired with them. That is why the PRIMARY is targeted-minus-blanket --
# both fresh, both CRN -- and the comparison against u=0 is secondary, read
# only for the direction and the magnitude of the blanket effect it should
# reproduce (+0.0589 on a broken constraint, already measured on S1b).
#
# RUNTIME ESTIMATE 65-90 min (--time 03:00:00 = ~2x the pessimistic end).
# ASSUMPTIONS, each with its source:
#   - 1104 branch points -> 2208 generations. MEASURED by the dry run, not
#     assumed: 48.4% of the 2280 branchable turns have >= 1 constraint broken
#     at t-1 (767 with one broken, 271 with two, 66 with all three).
#   - 1.6-2.2 s/generation. 1.61 is what THIS model, cap and context length
#     measured on the S1 arm (15772956, 9600 generations); 2.01 and 2.23 are
#     the branch and fidelity arms at batch 12. The branch points' mean turn is
#     11.7 against 11.0 for a uniform arm, so the prefix profile matches S1's
#     (median 7.7k reply tokens ahead of the branch, p90 19.9k, max 34.7k --
#     inside --max-model-len 49152, which S1 already ran at).
#   - + ~4 min model load and CUDA graph capture.
#   NOT MEASURED: whether a 1104-wide queue at this context length schedules as
#   well as S1's 40-wide one. --batch-size 128 bounds what one bad call costs.
#
# ARTIFACTS: outputs/sequor_targeted_branch_pilot/ does not exist (dry-run
# checked; the runner refuses to write into an existing directory). Rows are
# written before any summary -- job 15761810 lost 52 GPU-minutes of generation
# to a NameError raised while building its report.
#
# NO pip install here: the vLLM env loads shared modules by file path, so the
# 2026-09-08 concurrent editable-install race cannot recur.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_sequor_targeted_branch_pilot.py \
  --base-arm-dir outputs/sequor_s1_arm \
  --target-readout outputs/sequor_s1_arm/readout_independent_qwen3_14b.json \
  --agreement-readout outputs/sequor_s1_arm/readout_self_qwen3_4b_instruct_2507.json \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --max-new-tokens 2048 \
  --max-model-len 49152 \
  --batch-size 128 \
  --mode canonical \
  --out-dir outputs/sequor_targeted_branch_pilot
