#!/bin/bash
#SBATCH --job-name pdc-refusal-readout-expanded
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/refusal-readout-expanded-%j.out

# docs/experiments/adaptive_vs_fixed_claim_plan.md T3 Step 1 (section 5.1): expand the
# cached 4-arm activation-projection readout (320 rows, only 43 reminded transitions,
# input-gain test underpowered at p=0.0995) to 6 additional arms whose calibration-attack
# overlap is verified zero (see this plan's precheck, gate G3-1) -- 600 more rows / 480
# transitions / 286 reminded transitions, enough power to test whether u_remind actually
# moves the projection.
#
# Reuses the existing outputs/safety_direction_readout_heldout_excluded/ direction
# (calibrated excluding the 8 held-out attacks) -- does NOT recalibrate it, so the
# steering/readout artifacts every other experiment depends on stay untouched.
#
# --layer 18 must match that direction's calibration layer (the script itself asserts
# this). Two forward passes per row (before/after reply), no token generation.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true
python -m pip install -e . -q

python scripts/analyze_refusal_direction_readout.py \
  --arm phaseA_constant_remind=outputs/koopman_defense_phaseA_constant_remind \
  --arm phaseE_constant_remind=outputs/koopman_defense_phaseE_constant_remind \
  --arm phaseG_periodic=outputs/koopman_defense_phaseG_periodic \
  --arm phaseE_koopman_mpc=outputs/koopman_defense_phaseE_koopman_mpc \
  --arm phaseI_koopman_mpc_valigned=outputs/koopman_defense_phaseI_koopman_mpc_valigned \
  --arm phaseE_threshold=outputs/koopman_defense_phaseE_threshold \
  --direction-path outputs/safety_direction_readout_heldout_excluded/safety_direction.npy \
  --direction-stats-path outputs/safety_direction_readout_heldout_excluded/safety_direction_stats.json \
  --layer 18 \
  --out-path outputs/koopman_case_study/refusal_direction_readout_expanded.json
