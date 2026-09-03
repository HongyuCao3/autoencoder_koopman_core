#!/bin/bash
#SBATCH --job-name pdc-refusal-readout
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/refusal-readout-%j.out

# docs/task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 1's never-executed
# clause: "顺带确认 refusal 方向投影与 judge 分的相关性 (readout 有效性)". That document's
# readout list puts the activation projection FIRST and the LLM judge second as an
# external check; every experiment on this line has used only the judge.
#
# Worth doing now because the judge's y_safety takes 5 values (bin width 0.25) with
# 61% of Phase J's scores at the 1.00 ceiling, while the differences Phase J had to
# resolve were 1-2 judge bins across 16 trajectories
# (docs/experiments/koopman_defense_pilot.md, "读出与判据的检查"). The projection is
# deterministic, continuous, and recomputable from the stored agent_message without
# re-running a single conversation.
#
# Step 1 recalibrates the direction EXCLUDING the 8 held-out attacks. The existing
# outputs/safety_direction/ artifact (2026-08-31) overlaps all 8 of them, which is
# harmless for its original purpose (a channel-C steering actuator) and disqualifying
# for a readout, since the harmless pole is literally those attacks' turn-1 queries.
# Written to a separate directory so the steering artifact stays byte-identical --
# run_dose_response_screening.py's --direction-path still points at the old one.
#
# Step 2 projects every row of four Phase J arms at both positions (before the reply,
# which is where the direction was calibrated, and after it, which is what the
# feasibility doc asks for) and correlates each with the judge score.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true
python -m pip install -e . -q

HELD_OUT=safemtdata_0074,safemtdata_0169,safemtdata_0257,safemtdata_0289,safemtdata_0324,safemtdata_0329,safemtdata_0476,safemtdata_0530

python scripts/calibrate_safety_direction.py \
  --layer 18 \
  --exclude-attack-ids "$HELD_OUT" \
  --output-dir outputs/safety_direction_readout_heldout_excluded

python scripts/analyze_refusal_direction_readout.py \
  --direction-path outputs/safety_direction_readout_heldout_excluded/safety_direction.npy \
  --layer 18
