#!/bin/bash
#SBATCH --job-name pdc-indepjudge-rejudge-selfconsistency
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/indepjudge-rejudge-selfconsistency-%j.out

# Step 4 of docs/experiments/independent_judge_reactive_rerun_plan.md: rejudge
# the 5 reactive arms that were just rerun end-to-end with the independent
# judge (Qwen/Qwen3-4B-Instruct-2507). Re-scoring a file that is already
# independent-judge-scored with the same judge/seed convention should
# reproduce y_safety bit-for-bit -- that reproduction is a self-consistency
# check on the online judge call vs rejudge_safety_runs.py's offline path,
# and it also lands each arm's scores at the same relative path
# (rejudge_qwen3_4b_instruct_2507/trajectories.jsonl) that
# analyze_budget_arm_comparison.py's --scores-name expects for every arm.
#
# Only touches the 5 *_indepjudge arm dirs (explicit --arm-dir list) --
# does not re-touch the self-judged originals or the fixed arms' existing
# rejudge output.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/rejudge_safety_runs.py --device cuda \
  --arm-dir outputs/koopman_defense_phaseJ_budget1_koopman_indepjudge \
  --arm-dir outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge \
  --arm-dir outputs/koopman_defense_phaseE_threshold_indepjudge \
  --arm-dir outputs/koopman_defense_phaseE_koopman_mpc_indepjudge \
  --arm-dir outputs/koopman_defense_phaseI_koopman_mpc_valigned_indepjudge \
  --manifest-path outputs/koopman_case_study/rejudge_manifest_indepjudge.json
