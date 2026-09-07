#!/bin/bash
#SBATCH --job-name pdc-phaseJ-threshold-indepjudge-ymin1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseJ-threshold-indepjudge-ymin1-%j.out

# Phase J (threshold arm, independent judge, most-sensitive bang-bang y_min=1.0):
# docs/experiments/adaptive_vs_fixed_claim_plan.md T1a. Bounds the "starved
# controller" component of "koopman/threshold significantly worse under the
# independent judge" -- pushing y_min to the most sensitive setting the
# {0,0.25,0.5,0.75,1.0} readout allows, still under the independent judge.
# NOT meant to flip the conclusion back: the plan's section 0.1 already
# measured the trigger-rate ceiling at 1.6%-5.0% here, against 12.5%-13.8%
# for the self-judged y_min=0.7 baseline.
#
# Same 8 held-out attacks x 5 seeds as every other Phase J arm. Brand-new
# output_dir, so no `allow_config_change=true` -- nothing to resume into.
#
# Driven through scripts/run_screening_hydra.py (not run_defended_screening.py):
# the arm's controller settings AND its output_dir come from one
# conf/experiment/*.yaml file, and persona_drift.run_config_guard refuses to
# resume into a directory that recorded a different config.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_screening_hydra.py experiment=phaseJ_indepjudge_threshold_ymin1
