#!/bin/bash
#SBATCH --job-name pdc-phaseJ-fixed-t1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseJ-fixed-t1-%j.out

# Phase J (fixed schedule, turn 1): budget-constrained re-run of the adversarial-defense control
# problem -- at most ONE reminder per 5-turn trajectory
# (docs/experiments/budget_constrained_defense_plan.md, from docs/next_step_diagnosis.md
# section 4 step 2). The same 8 held-out attacks as Phase E/G/H/I, now x 5 seeds
# (docs/next_step_diagnosis.md section 4 step 3): the 2-seed run could not
# separate adjacent arms, so section 10.7 of the plan doc made this expansion a
# precondition for anything further on this line. `allow_config_change=true`
# below is for exactly that one changed key -- trajectories are keyed
# `<attack_id>__seed<n>`, so the 16 already-collected ones are skipped and only
# the 24 new seeds are generated (a resume, not a re-run); the 2-seed reports
# are preserved under outputs/<arm>/snapshot_2seed/ before this overwrites them.
#
# Fixed-allocation baseline: always spend the single reminder on turn 1,
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
python scripts/run_screening_hydra.py experiment=phaseJ_budget1_fixed_t1 allow_config_change=true
