#!/bin/bash
#SBATCH --job-name pdc-sequor-s3-arm
#SBATCH --cpus-per-task 8
#SBATCH --mem 96G
#SBATCH --time 10:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s3-arm-%j.out

# S3 CLOSED LOOP (method: closed-loop evaluation + baseline comparison).
# NOT SUBMITTED WITHOUT A USER RULING (.claude/experiments.md).
#
# WHAT IT DECIDES: RQ3, the paper's closed-loop table. Does a controller that
# READS the state beat the best schedule that cannot, at equal token cost?
#
# WHY IT IS ALLOWED TO RUN AT ALL. The arm died on paper three times, all with
# a single on/off action: the best SCHEDULE was identical for every trajectory
# under a scalar operator (+0.0000024), under a per-constraint binary state
# (+0.0003), and under a priced threshold objective (+0.0084). The targeted
# branch arm (15792569) changed the ACTION and the degeneracy broke: naming
# only the broken constraints recovers them at +0.0748 over naming all three
# (1.47x MDE, CI [+0.0367, +0.1202]), and the size of that gain depends on HOW
# MANY are broken -- +0.1489 at one, +0.0018 at two, exactly +0.0000 at three,
# where the targeted block IS the blanket block. An action whose value depends
# on a state that is not knowable in advance is the thing no fixed schedule can
# imitate, and it is the only reason this arm is being run.
#
# THREE ARMS, ONE PER-ITEM TOKEN BUDGET (15% of the full-dose blanket spend):
#   koopman_mpc          state + fitted model + budget-aware plan
#   greedy_targeted      state, no model: name whatever is broken
#   best_fixed_schedule  open-loop, the best schedule the model can find
# The open-loop arm plays only `none` or `blanket` because "name the broken
# ones" is undefined without reading the state. That asymmetry is the
# hypothesis, not a starved baseline.
#
# greedy_targeted is NOT in the plan's original three arms. Without it, a
# positive MPC result cannot separate "the fitted operator earned this" from
# "reacting at all earned this", which is the first question a reader asks.
#
# `equal_cost_random` IS in the plan's original three and is CUT (user ruling,
# 2026-09-12). What the cut costs, stated so nobody reads the missing arm as an
# oversight: it was the one contrast the model predicted would resolve
# comfortably (+0.0862, 1.40x MDE), so this job gives up its most likely
# positive result. What it buys is 3,000 generations and ~2 hours. It does NOT
# buy precision -- the headline MDE is set by the number of dialogues and is
# 0.0617 either way. It was chosen for the cut because "a planned controller
# beats a random one" is the least contested claim on the table, and because
# `best_fixed_schedule` is the open-loop competitor a reader actually argues
# with.
#
# ZERO CONTROL IS NOT REGENERATED: outputs/sequor_s1_arm/'s zero_control rows
# are the same items, harness and seeds at u=0 throughout.
#
# PRE-REGISTERED ESTIMAND (signed before submission):
#   PRIMARY    late-window (t15..t20) mean `y` on the INDEPENDENT judge,
#              koopman_mpc minus best_fixed_schedule, paired by (item, seed),
#              bootstrapped over items.
#   SECONDARY  minus greedy_targeted (what the operator is worth beyond
#              reacting) and minus the stored zero_control. Reported with the
#              primary, never instead of it.
#   MDE        0.0617 at 50 items x 3 seeds, carried from S1's own paired
#              late-window variance components (G-S2-4, signed 2026-09-10).
#   CI containing 0 -> UNDECIDABLE (screening section 10 item 10), never
#   "the closed loop is worthless".
#
# WHAT THE MODEL PREDICTS, AND WHY THAT IS NOT A REASON TO SKIP IT. The fitted
# model puts koopman_mpc - best_fixed_schedule at +0.0531 at this budget --
# 0.86x the MDE, i.e. this design is a coin flip on its own headline. It is
# submitted anyway because the same class of model under-called this line's
# measured endpoint by 2.8x (the operator's steady state 0.050 against S1a's
# +0.1389), and because the budget curve it predicts is itself the result:
# the closed loop's advantage is concentrated where reminders are scarce
# (+0.0589 at 10% of full dose, +0.0531 at 15%, +0.0237 at 35%, +0.0141 at
# 50%) and vanishes when the budget stops binding. Reporting that curve needs
# one measured point on it.
#
# RUNTIME ESTIMATE 4.3-7.0 h (--time 10:00:00 = ~1.4x the pessimistic end).
# ASSUMPTIONS, each with its source:
#   - 3 arms x 50 items x 20 turns x 3 seeds = 9,000 generations. 1.4-2.0
#     s/generation: 1.61 is what this model, cap and context length measured on
#     the S1 arm (15772956, 9600 generations at batch 40); this arm batches 50.
#     -> 3.5-5.0 h.
#   - 27,000 in-loop judge calls (3 per generation), in 180 batches of 150.
#     0.099 s/call is the 4B judge's measured rate (15739196) at batch 2000;
#     at batch 150 it will be worse. -> 0.8-1.9 h.
#   - + ~4 min model load. ONE model serves both roles: the in-loop judge IS
#     the agent (plan section 8), so nothing is loaded twice.
#   NOT MEASURED: how vLLM schedules alternating long-context generation and
#   short judge prompts in the same process. Rows are written before any
#   summary, so a timeout loses the tail and not the run.
#
# ARTIFACTS: outputs/sequor_s3_arm/ does not exist (dry-run checked; the runner
# refuses to write into an existing directory).
#
# NO pip install here: the vLLM env loads shared modules by file path.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/run_sequor_s3_arm.py \
  --action-model outputs/sequor_s3_design/action_model_fit.json \
  --agent-model Qwen/Qwen3-4B-Instruct-2507 \
  --n-items 50 \
  --n-turns 20 \
  --late-from 15 \
  --seeds 0 1 2 \
  --arms koopman_mpc greedy_targeted best_fixed_schedule \
  --budget-share 0.15 \
  --constraints-in-system \
  --decoding model_default \
  --max-new-tokens 2048 \
  --inloop-max-new-tokens 1024 \
  --max-model-len 49152 \
  --mode canonical \
  --out-dir outputs/sequor_s3_arm
