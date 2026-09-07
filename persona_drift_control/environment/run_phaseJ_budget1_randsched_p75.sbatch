#!/bin/bash
#SBATCH --job-name pdc-phaseJ-randsched-p75
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseJ-randsched-p75-%j.out

# Phase J equal-cost random-allocation baseline, spend_prob=0.75
# (docs/experiments/adaptive_vs_fixed_claim_plan.md T2): 0.75 is
# `phaseJ_budget1_koopman`'s actual self-judge spend rate (30 reminders / 40
# trajectories), so this arm is genuinely equal-cost against koopman_b1,
# unlike the p100 arm above or any of the fixed_t1..5 arms (which always
# spend 1.000).
#
# Driven through scripts/run_screening_hydra.py (not run_defended_screening.py):
# the arm's controller settings AND its output_dir come from one
# conf/experiment/*.yaml file, and persona_drift.run_config_guard refuses to
# resume into a directory that recorded a different config.
#
# Brand-new output directory (verified not to exist before submission), so no
# allow_config_change=true -- there is nothing to resume into.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_screening_hydra.py experiment=phaseJ_budget1_randsched_p75
