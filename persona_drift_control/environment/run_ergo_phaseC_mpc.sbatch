#!/bin/bash
#SBATCH --job-name pdc-ergoC-mpc
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoC-mpc-%j.out

# F3 tested arm: ErgoKoopmanMPCController on the F2 closeness fit
# (koopman_fit_report_closeness.json), budget 1, shard_frac truth-overridden
# during lookahead, episode_length taken per-trajectory from num_shards
# (all handled inside ErgoKoopmanMPCController -- see
# src/persona_drift/ergo_koopman_mpc.py and
# docs/experiments/signal_resolution_plan.md section 4.1).

set -euo pipefail
module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
nvidia-smi -L || true
python -m pip install -e . -q

ITEM_IDS=$(cat conf/experiment/ergo_phaseC_item_ids.txt)
python scripts/run_ergo_math_screening.py \
  --agent-model Qwen/Qwen3-4B --device cuda \
  --output-dir outputs/ergo_math_phaseC_mpc \
  --controller ergo_koopman_mpc \
  --koopman-model-path outputs/ergo_math_phaseB_random_excite/koopman_fit_report_closeness.json \
  --koopman-model-key arx --koopman-nu 1 --koopman-mu 1 --koopman-horizon 2 \
  --koopman-repeat-penalty 0.0 --koopman-y-col closeness --remind-budget 1 \
  --item-ids $ITEM_IDS \
  --seeds 0 1
