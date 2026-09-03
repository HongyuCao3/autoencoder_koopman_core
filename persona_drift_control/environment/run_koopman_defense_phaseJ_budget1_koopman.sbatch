#!/bin/bash
#SBATCH --job-name pdc-phaseJ-koopman
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/phaseJ-koopman-%j.out

# Phase J (koopman arm): budget-constrained re-run of the adversarial-defense control
# problem -- at most ONE reminder per 5-turn trajectory
# (docs/experiments/budget_constrained_defense_plan.md, from docs/next_step_diagnosis.md
# section 4 step 2). Same 8 held-out attacks x 2 seeds as Phase E/G/H/I.
#
# The adaptive arm: Phase I's v-aligned state-action-interaction model
# (interaction_model_report_valigned.json, nu=1, mu=2, repeat_penalty=0),
# re-solved under the budget with horizon=5/episode_length=5 so it can
# actually represent "spend the last reminder now or save it for a turn I
# predict will be worse", and pad_short_history=true so it is allowed to
# act from turn 2 like periodic can. Offline replay before submitting
# (scripts/analyze_budget_allocation_replay.py) confirmed the placement is
# genuinely trajectory-dependent (turns 3/4/5, plus 3-5/16 that keep the
# reminder unspent) rather than degenerating into a fixed schedule -- that
# check is why this GPU run is worth submitting at all.
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
python scripts/run_screening_hydra.py experiment=phaseJ_budget1_koopman
