#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseH-interaction
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseH-interaction-%j.out

# Phase H (docs/experiments/koopman_case_study_design.md's "后续: 验证'对下
# 一步方向的启示'"): the first real closed-loop test of the state-action
# interaction fix identified there. Offline replay (analyze_state_action_
# interaction.py) already confirmed the interaction-augmented model +
# repeat_penalty=0.2 produces a genuine 26/32 mixed decision on Phase E's
# recorded states (not the uniform 32/32/0/32 either extreme produces) --
# this job is the actual live run, since offline replay can only show what
# the policy WOULD decide given states generated under a DIFFERENT policy
# (repeat_penalty=0), not what really happens once the different reminder
# pattern changes the model's real responses downstream.
#
# --koopman-repeat-penalty 0.2: chosen as a middle point in the offline
# sweep (0.1->31/32, 0.15->30/32, 0.2->26/32, 0.25->14/32) -- meaningful
# selectivity without the more aggressive pruning at 0.25, given the
# documented open concern that this model's margin correlates POSITIVELY
# with y_probe (repeat_penalty prunes reminders on the WORSE states first,
# opposite of naive intuition) -- see the doc's "一个需要如实指出、目前还
# 没有答案的问题" section.
#
# Same 8 held-out attacks as every other Phase E/G arm (never in Phase B/C's
# fitting data) -- these attacks are NOT homogeneous (see the same doc):
# safemtdata_0074 crashes to y_safety=0.0 by turn 3 under zero_control,
# while safemtdata_0169/0530 never drop below 0.75 -- exactly the kind of
# heterogeneity a real adaptivity advantage over `periodic` would need to
# show up on.
#
# --koopman-interaction-model-path defaults to
# outputs/koopman_case_study/interaction_model_report.json
# (analyze_state_action_interaction.py's fit, repeat_penalty is a runtime
# controller parameter, not part of that saved fit).

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
  --output-dir outputs/koopman_defense_phaseH_koopman_mpc_interaction \
  --seeds 0 1 \
  --controller koopman_mpc_interaction \
  --koopman-nu 1 --koopman-mu 2 --koopman-horizon 2 --koopman-repeat-penalty 0.2 \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
