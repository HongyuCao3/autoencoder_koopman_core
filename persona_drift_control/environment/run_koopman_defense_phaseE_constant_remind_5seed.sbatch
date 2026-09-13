#!/bin/bash
#SBATCH --job-name pdc-defense-const-5seed
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/defense-const-5seed-%j.out

# G1 of docs/article/MAIN_TABLE_DESIGN.md section 5: Table 2's `defense` column needs its two
# endpoint arms -- zero_control (no defense) and constant_remind (full dose) -- at the
# >=3-seed reporting caliber .claude/global.md requires. Every Phase J budget-1 arm already
# has 5 seeds; these two, run at Phase E time (2026-08-31, jobs 15414045 / 15414046), have 2.
#
# WHY THIS IS A FRESH 5-SEED RUN AND NOT THE RESUME THE DESIGN DOC PENCILLED IN.
# The design doc's G1 row says "续跑" (resume seeds 2-4 into the existing Phase E directory),
# which is what Phase J did on 2026-09-03 for its own arms. The difference: Phase J resumed
# the SAME DAY into directories that record a config snapshot (hydra_run_config.json, guarded
# by persona_drift.run_config_guard). The Phase E directories were written by this CLI runner,
# carry no config snapshot, and are 12 days old. The generation path was read line by line
# across that gap and is behaviour-preserving -- the turn loop moved into
# trajectory_runner.run_reminder_gated_trajectory with agent_seed = seed*1e6 + turn*100 + 1
# and judge_seed offset 2 unchanged (commit b135119 / a86998a), safety_judge's 1-5 parse moved
# verbatim into judge_scoring.parse_1_to_5_score, chat_model.generate's prompt construction was
# factored out without a semantic edit, and the stored rows confirm the decoding config is the
# same on both sides (max_new_tokens 256 / temperature 0.7 / top_p 0.95, Qwen3-4B self-judge).
# What CANNOT be verified after the fact is the Python environment itself: torch/transformers
# in /scratch/hcao2/envs/persona_drift_pilot may have moved since 2026-08-31, and a resume
# would fold that unknown into the cross-seed std of a reported `mean ± std (n=5)` with no way
# to see it. Generating all 5 seeds here costs 16 extra trajectories (about 11 min) and buys
# two things: every seed in the Table 2 cell carries one harness fingerprint, and seeds 0/1
# become a direct, zero-GPU measurement of whether the instrument drifted (compare against
# outputs/koopman_defense_phaseE_constant_remind/trajectories.jsonl, byte-for-byte on
# agent_message). The existing Phase E artifacts are NOT touched -- new output directory,
# nothing overwritten (.claude/global.md -> 产物与谱系: outputs/ 只增不改).
#
# Same 8 held-out attacks as every Phase E/F/G/H/I/J arm (Phase C's identification split), so
# this stays comparable to the budget-1 arms it will sit beside in Table 2.
#
# Runtime estimate 30 +/- 5 min: 8 attacks x 5 seeds = 40 trajectories x 5 turns, at the 41 s
# per trajectory job 15414046 actually measured (16 trajectories in 625 s of generation, plus
# ~1 min model load), on the same A100 and the same Qwen3-4B at max_new_tokens=256. Assumption
# not re-measured: that an 80GB vs 40GB a100 allocation does not change per-trajectory latency
# here (batch width is 1 in this runner, so it should not). --time 01:30:00 is 3x the estimate.
#
# The pip install is serialized with flock: two jobs installing this package into the shared
# env at once lose __editable__.persona_drift_control-0.1.0.pth and die inside a minute
# (2026-09-08, jobs 15696222 / 15696227).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

mkdir -p /scratch/hcao2/envs/.locks
flock /scratch/hcao2/envs/.locks/persona_drift_pilot.editable.lock python -m pip install -e . -q

python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseE_constant_remind_5seed \
  --seeds 0 1 2 3 4 \
  --controller constant_remind \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
