#!/bin/bash
#SBATCH --job-name pdc-zerocontrol-proj
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/zerocontrol-proj-%j.out

# Closes the last open direction left by docs/experiments/adaptivity_ceiling_results.md.
#
# That analysis bounds every policy measurable w.r.t. the y_safety readout at
# +0.0063 (CI crosses zero) and names the mechanism: at the turn-1 and turn-2
# decisions, all 40 trajectories look identical to a y-conditioned controller
# (y_1 == 1.0 everywhere). It explicitly does NOT bound a controller reading
# activations, and the zero-GPU screen of seven text/metadata signals closed
# only the cheap half of that opening.
#
# WHY THIS ARM AND ONLY THIS ARM. The observable history before the controller
# fires IS the zero_control prefix -- fixed_t{k} is byte-identical to
# zero_control for turns 1..k-1 (guard_prefix_identity, 400/400 rows). The
# OUTCOME of firing at turn t is fixed_t{t}'s y_safety, which is already on
# disk. So only the conditioning signal is missing, and it is missing on
# exactly one arm. 200 rows, one forward pass each, NO generation.
#
# WHAT THE 2-SEED PILOT ALREADY SAID. outputs/koopman_case_study/
# refusal_direction_readout.json carries proj_pre_reply for phaseJ_fixed_t4,
# whose turns 1-3 are unfired, i.e. 16 trajectories of zero-control prefix.
# On those: proj separates 8 classes at the turn-2 decision where y separates
# 1, and the in-sample oracle rises to +0.0469 (CI excludes zero) against y's
# +0.0156 -- so the signal is real. But leave-one-attack-out it is -0.0417,
# i.e. it does not transfer to an attack never seen. This job exists to settle
# that on 40 trajectories rather than 16, and to add the two things the pilot
# structurally cannot have: the turn-5 decision point (needs turn-4 unfired
# projections, absent from fixed_t4) and a meaningful leave-one-SEED-out axis,
# which is the only deployment claim under which conditioning on attack
# identity is legitimate and where the per-attack oracle's +0.0375 lives.
#
# The direction is the held-out-excluded artifact (2026-09-03, layer 18, zero
# overlap with the 8 scored attacks). It is NOT recalibrated here: recomputing
# it would produce a second artifact competing with the one the pilot numbers
# came from. The script refuses an overlapping direction on its own.
#
# Runtime: ~2 min. Basis -- the same script did 320 rows in 1m53s (job
# 15506111) and 600 rows in 2m12s (15617737), both dominated by model load;
# this is 200 rows. Qwen3-4B, a100:1, no generation. --time is 00:30:00 = 15x.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true
python -m pip install -e . -q

python scripts/analyze_refusal_direction_readout.py \
  --arm zero_control_5seed=outputs/koopman_defense_phaseE_zero_control_5seed \
  --direction-path outputs/safety_direction_readout_heldout_excluded/safety_direction.npy \
  --direction-stats-path outputs/safety_direction_readout_heldout_excluded/safety_direction_stats.json \
  --layer 18 \
  --out-path outputs/koopman_case_study/zero_control_projection_readout.json
