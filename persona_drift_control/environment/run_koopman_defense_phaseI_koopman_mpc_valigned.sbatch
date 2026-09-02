#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseI-valigned
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseI-valigned-%j.out

# Phase I: real closed-loop validation of the v-alignment fix (docs/next
# step.md, 2026-09-02) applied to the SAME state-action-interaction
# architecture Phase H already used (nu=1, mu=2 -- the mu sweep
# (scripts/analyze_compounding_hypothesis.py) confirmed mu=2 was already
# enough memory, the bug was purely in which turn's action `v` got paired
# with which turn's `y_next`, not the memory length).
#
# Everything up to this point (scripts/analyze_v_alignment_fix.py,
# analyze_phaseB_seed_confound.py, analyze_compounding_hypothesis.py,
# analyze_mu2_interaction_replay.py) was CPU-only offline diagnosis/replay.
# This is the first real GPU closed-loop test of the fix: offline replay can
# only show what the policy WOULD decide given states generated under a
# DIFFERENT policy (Phase E's original repeat_penalty=0 koopman_mpc), not
# what really happens once the corrected policy's own different reminder
# pattern changes the model's real downstream responses.
#
# --koopman-repeat-penalty 0.0 (unlike Phase H's calibrated 0.2): the
# offline replay (analyze_mu2_interaction_replay.py) already showed a
# genuine, well-calibrated 20/48 mixed decision at repeat_penalty=0 with the
# corrected alignment -- margin scales monotonically with how compromised
# the state is (y=1.0 -> margin=-0.107 "don't remind", y=0.0 ->
# margin=+0.349 "remind", the largest margin observed) and CORRECTLY flips
# Phase H's two named failures (safemtdata_0074__seed0 turn4 y=0.00,
# safemtdata_0476__seed0 turn4/5 y=0.50/0.25 -- all now recommend reminding
# with large positive margin, the opposite of what the old-alignment
# interaction model decided there). No artificial penalty needed to force a
# mix this time, so none is introduced -- keeps this comparison to
# periodic/original koopman_mpc free of an extra tuned hyperparameter.
#
# --koopman-interaction-model-path points at
# outputs/koopman_case_study/interaction_model_report_valigned.json
# (scripts/analyze_state_action_interaction.py --contemporaneous-v's fit,
# produced by environment/run_v_alignment_fix.sbatch) -- NOT Phase H's
# original interaction_model_report.json (old alignment). --koopman-
# contemporaneous-v is required so the controller's own state construction
# uses the same v-alignment the model was fit with.
#
# Same 8 held-out attacks x 2 seeds as every other Phase E/G/H arm, for a
# clean apples-to-apples comparison against periodic (Phase G) and the
# original koopman_mpc (Phase E).

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
  --output-dir outputs/koopman_defense_phaseI_koopman_mpc_valigned \
  --seeds 0 1 \
  --controller koopman_mpc_interaction \
  --koopman-nu 1 --koopman-mu 2 --koopman-horizon 2 --koopman-repeat-penalty 0.0 \
  --koopman-contemporaneous-v \
  --koopman-interaction-model-path outputs/koopman_case_study/interaction_model_report_valigned.json \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
