#!/bin/bash
#SBATCH --job-name pdc-phaseJ-fixed-t2
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseJ-fixed-t2-%j.out

# Phase J (fixed schedule, turn 2): budget-constrained re-run of the adversarial-defense control
# problem -- at most ONE reminder per 5-turn trajectory
# (docs/experiments/budget_constrained_defense_plan.md, from docs/next_step_diagnosis.md
# section 4 step 2). Same 8 held-out attacks x 2 seeds as Phase E/G/H/I.
#
# Fixed-allocation baseline: always spend the single reminder on turn 2,
# no feedback. The full turn-1..5 sweep is run because the claim under test
# is "the adaptive policy beats the BEST fixed allocation", and which turn
# that is can only be known by measuring it; the sweep is also a result on
# its own (where a single reminder pays off most against these attacks).
#
# Driven through scripts/run_screening_hydra.py (not run_defended_screening.py):
# the arm's controller settings AND its output_dir come from one
# conf/experiment/*.yaml file, and persona_drift.run_config_guard refuses to
# resume into a directory that recorded a different config -- the screening
# loop resumes silently otherwise, which with several near-identical arms is
# the realistic way to end up with one report averaging two controllers.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_screening_hydra.py experiment=phaseJ_budget1_fixed_t2
