#!/bin/bash
#SBATCH --job-name pdc-sequor-judge-smoke
#SBATCH --cpus-per-task 8
#SBATCH --mem 48G
#SBATCH --time 00:30:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-judge-smoke-%j.out

# S0 SMOKE TEST (infrastructure, not evidence -- the report it writes carries
# mode="debug" and --limit, and docs/experiments/constraint_retention_plan.md
# section 8 forbids debug artifacts from being cited).
#
# WHAT IT BUYS: the one number the S0 runtime estimate is missing -- seconds
# per judged row. Upstream's judge prompt (vendored verbatim in
# sequor_constraint_judge.py) explicitly invites the judge to reason before
# emitting "Final Verdict: [[Yes/No]]", so output length is NOT the 10-token
# affair this project's other judges are, and the full 2000-row calibration is
# somewhere between 40 minutes and 11 hours per candidate depending on it.
# 24 rows at 100% of the real prompt shape settles that before anything large
# is queued. It also measures the parse-failure and token-cap rates, which
# decide whether --max-new-tokens 512 is enough for this model.
#
# 24 rows spans both labels and both source models: --limit takes a STRATIFIED
# slice (6 per (source_model, label) cell), not a head slice. The gold file is
# ordered by cell, so rows[:24] would have been 24 gpt-5.2 `follow` rows and
# nothing else -- the dry run caught that, and it is now a test
# (tests/test_sequor_judge_calibration.py). The accuracy numbers off 24 rows
# are still NOT calibration; timing, parse-failure rate and token-cap rate are
# what this run is for.
#
# The pip install is serialized with flock (2026-09-08: two concurrent editable
# installs into this env killed jobs 15696222 / 15696227 inside a minute).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

mkdir -p /scratch/hcao2/envs/.locks
flock /scratch/hcao2/envs/.locks/persona_drift_pilot.editable.lock python -m pip install -e . -q

python scripts/calibrate_sequor_judge.py \
  --judge-model Qwen/Qwen3-14B \
  --device cuda \
  --max-new-tokens 512 \
  --limit 24 \
  --out-path outputs/sequor_s0_smoke/judge_calibration_qwen3_14b_smoke.json
