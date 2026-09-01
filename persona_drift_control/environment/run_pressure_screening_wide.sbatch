#!/bin/bash
#SBATCH --job-name pdc-pressure-screening-wide
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 06:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/pressure-screening-wide-%j.out

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q

# Wider-N follow-up to pressure_screening_pilot.md's job 15406535 (4 prompts,
# 2 per category, df=3): that run found escalating_pressure trending the
# predicted direction (3/4 negative slopes) but underpowered to reach
# significance (t=-1.2847, p=0.2891). This variant only changes --num-prompts
# 4 -> 12 (6 per category, within the character_traits pool's 13-entry cap
# with margin, see prompt_bank.load_prompt_bank) to raise df from 3 to 11 --
# everything else (seeds, turns, probe-repeats, model) is unchanged so the
# two runs' t/p values are directly comparable. Writes to a separate
# --output-dir so job 15406535's result isn't overwritten.
#
# Note: select_screening_prompts(rng_seed=0, num_prompts=12) is not
# guaranteed to be a superset of the num_prompts=4 draw (random.sample's
# output depends on k) -- this is a fresh, larger sample from the same pool,
# not an extension of the original 4 prompts specifically.
#
# --time is scaled from the original run's measured 5087s/16 trajectories
# (~318s/trajectory average) to 48 trajectories (12 prompts x 2 seeds x 2
# conditions): ~15260s (~4.24h) estimated, budget below leaves ~40% margin.
# Supports resumption like screening.py -- if this hits --time, resubmit
# this same sbatch script.
python scripts/run_pressure_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --num-prompts 12 --seeds 0 1 --prompt-rng-seed 0 \
  --output-dir outputs/pressure_screening_wide
