#!/bin/bash
#SBATCH --job-name pdc-rejudge-d1-qwen4b
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/defense-rejudge-d1-qwen4b-%j.out

# K1.2 of docs/experiments/koopman_fast_track_plan.md: offline independent
# rejudge of the ALREADY-COLLECTED outputs/d1_screen_qwen4b/. No new generation
# -- rejudge_safety_runs.py only re-scores existing agent_message text and
# writes to a new sub-directory, so the original trajectories.jsonl is never
# touched.
#
# One question only: does the independent judge have RANGE on this setting?
# On the old pool it pinned 75/80 rows at exactly 1.0, with turns 1/2/4 at
# sd == 0 -- which is why that arm's "lag-1 = 0.000, p=1" had zero informative
# pairs and could not have come out any other way. If the independent judge
# moves here, K4 reports both judges side by side; if it saturates again, K4
# reports the self-judge alone and carries section 14.1's limitation verbatim.
#
# Sets no gate and blocks nothing. Runs in parallel with K1.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

# The shared env at /scratch/hcao2/envs/persona_drift_pilot is ONE directory for
# every job, and `pip install -e .` rewrites __editable__*.pth in it. Two jobs
# submitted together therefore race and one dies with
# "OSError: [Errno 2] ... __editable__.persona_drift_control-0.1.0.pth"
# (that is exactly how job 15696222 failed on 2026-09-08). flock serialises it.
mkdir -p /scratch/hcao2/envs/.locks
flock /scratch/hcao2/envs/.locks/persona_drift_pilot.editable.lock \
  python -m pip install -e . -q
python scripts/rejudge_safety_runs.py --device cuda \
  --arm-dir outputs/d1_screen_qwen4b \
  --manifest-path outputs/koopman_case_study/rejudge_manifest_d1_screen_qwen4b.json
