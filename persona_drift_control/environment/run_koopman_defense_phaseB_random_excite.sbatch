#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseB
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseB-%j.out

# Phase B of docs/experiments/koopman_defense_pilot.md: open-loop excitation
# data collection for Koopman identification (--controller random_excite,
# p=0.5 -- each turn independently 50/50 whether the safety reminder is
# inserted). Submitted in parallel with Phase A (job 15413205) rather than
# waiting for its gate result: the user accepted the small risk of wasted
# GPU time if Phase A shows the reminder actuator has no authority at all,
# in exchange for not losing ~30-40 min of wall-clock waiting on Phase A.
#
# --attack-rng-seed 100 (not 0, the baseline/Phase A seed) and --num-attacks
# 30 (wider than Phase A's 20): reduces (does not guarantee zero) overlap
# with Phase A/baseline's attack panel and with whatever attacks Phase E's
# held-out validation set ends up using -- same convention already used
# elsewhere in this codebase (dose_response_pilot.md's calibration/screening/
# dose-response rng seeds 100/0/200), documented there as a known,
# accepted limitation rather than a fully leak-proof split. If Phase E later
# needs a stricter guarantee, revisit with an explicit attack_id allowlist.
#
# --time estimated the same way as Phase A: ~35s/trajectory x 60
# trajectories (30 attacks x 2 seeds) =~ 35min, generous margin below.
# Resumable (adversarial_screening.py) -- a --time kill just needs `sbatch`
# on this same script again.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_defended_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/koopman_defense_phaseB_random_excite \
  --num-attacks 30 --seeds 0 1 --attack-rng-seed 100 \
  --controller random_excite --random-excite-p 0.5
