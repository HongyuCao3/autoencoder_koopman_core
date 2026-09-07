#!/bin/bash
#SBATCH --job-name pdc-phaseB-rejudge
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseB-rejudge-%j.out

# T1b Step 1 of docs/experiments/adaptive_vs_fixed_claim_plan.md: Phase B
# (koopman_defense_phaseB_random_excite, the identification data the
# self-judged surrogate model was fit on) has never been rejudged with the
# independent judge -- the 2026-09-03 rejudge batch only covered 10 arms
# (7 Phase J + zero_control + periodic + phaseI), not Phase B. Without an
# independent-judge score on the identification data, T1b's refit (fitting
# the surrogate on independent-judge scores instead of self-judged ones)
# cannot proceed. Modeled on environment/run_indepjudge_rejudge_selfconsistency.sbatch.
#
# Only touches this one arm dir; the original trajectories.jsonl is never
# modified (rejudge_safety_runs.py writes to a new out-subdir).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/rejudge_safety_runs.py --device cuda \
  --arm-dir outputs/koopman_defense_phaseB_random_excite \
  --manifest-path outputs/koopman_case_study/rejudge_manifest_phaseB.json
