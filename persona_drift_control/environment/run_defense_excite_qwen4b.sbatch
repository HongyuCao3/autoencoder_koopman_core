#!/bin/bash
#SBATCH --job-name pdc-excite-qwen4b
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 08:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/defense-excite-qwen4b-%j.out

# K1 of docs/experiments/koopman_fast_track_plan.md: the open-loop excitation
# arm on the d1 attack pool. This is NOT a pre-experiment -- it is the Koopman
# training set, and it is simultaneously the only way to (a) estimate B with
# real power (~250 reminded turns, vs D1's 41 rows/arm) and (b) separate A from
# a static per-episode offset. Every earlier "is there a state" test on this
# line ran on a zero-control arm, where A and that offset are not identifiable
# and B is not identifiable at all -- that is the root cause the fast-track
# plan's section five documents.
#
# Setting matched to outputs/d1_screen_qwen4b/ turn for turn: same 100 attacks
# (conf/experiment/d1_attack_ids.txt), same Qwen3-4B, same
# --agent-max-new-tokens 1024, seed 0. That arm is therefore the u == 0
# counterpart of this one, free of charge, and K3's zero_control seed 0 too.
#
# --random-excite-p 0.5 (user ruling, 2026-09-08): maximises B's power. It does
# mean ~half the turns carry a reminder, which is not the deployment regime --
# that is fine, this arm is for identification, not for reporting a defense
# rate. New output dir; overwrites nothing.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q

# --attack-ids is nargs="+" (scripts/run_defended_screening.py:84) and
# d1_attack_ids.txt is one whitespace-separated line of 100 ids, so $(cat ...)
# expands correctly. Deliberately NOT run through scripts/run_d1_screening.py:
# that CLI is zero-control only (no controller_factory override).
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/defense_excite_qwen4b \
  --controller random_excite \
  --random-excite-p 0.5 \
  --attack-ids $(cat conf/experiment/d1_attack_ids.txt) \
  --seeds 0 \
  --agent-max-new-tokens 1024
