#!/bin/bash
#SBATCH --job-name pdc-ergoEKA-entropy
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:45:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoEKA-entropy-%j.out

# Per-turn token entropy on the EK-A counterfactual-branch arm's BASE
# trajectories, so premise (iii) can be re-asked on the state axis the
# upstream literature actually uses (ERGO triggers a reset on entropy spikes,
# not on task progress). The counterfactual pairs already carry the ground
# truth for "did a reset help at this turn"; this job supplies the candidate
# state to pair it with. Zero new generation -- one teacher-forced forward
# pass over already-recorded text, deterministic given frozen weights.
#
# WHY --rows-path IS THE BASE FILE, NOT counterfactual_pairs.jsonl:
# the state must be read at turn k-1, which is a base-branch row. The
# counterfactual rows are single-turn records with their own trajectory_id
# and are recorded under reset_mode="append", so replaying them here would
# rebuild the wrong context twice over. The script's guard refuses them.
#
# --prompt-profile upstream is REQUIRED (added 2026-09-08): this arm ran under
# the upstream profile, whose history starts with a system message. Without
# the flag the replay drops it and the entropies come off a context the model
# never saw. The script also refuses to run if the rows disagree with the flag.
#
# RUNTIME: 759 rows / 135 trajectories, one forward pass each. The prior
# entropy job (15644575) did 666 legacy-profile rows in 1m29s wall INCLUDING
# model load; these rows are longer (mean 53 completion words vs legacy's
# near-empty replies, and up to 12 accumulated turns of context), so estimate
# 5-10 minutes. --time 00:45:00 is >=4x that. Assumptions: a100, Qwen3-4B,
# no re-sampling.
#
# The pip install is serialized with flock: two jobs installing this package
# into the shared env at once lose __editable__.persona_drift_control-0.1.0.pth
# and die inside a minute (2026-09-08, jobs 15696222 / 15696227).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

mkdir -p /scratch/hcao2/envs/.locks
flock /scratch/hcao2/envs/.locks/persona_drift_pilot.editable.lock python -m pip install -e . -q

python scripts/analyze_ergo_entropy_readout.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --prompt-profile upstream \
  --rows-path outputs/ergo_ekA_branch/trajectories.jsonl \
  --out-path outputs/ergo_ekA_branch/entropy_readout.json
