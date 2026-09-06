#!/bin/bash
#SBATCH --job-name pdc-ergo-authority60-reset
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergo-authority60-reset-%j.out

# Sample-expansion follow-up, reset arm -- see the sibling
# *_zero_control_60items.sbatch for full context. SAME 60 items as that
# sibling, so the two runs are a directly comparable paired sample.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_ergo_math_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/ergo_math_authority_reset_60items \
  --seeds 0 1 \
  --item-ids ergo_GSM8K_40 ergo_GSM8K_1207 ergo_GSM8K_825 ergo_GSM8K_621 ergo_GSM8K_44 \
    ergo_GSM8K_346 ergo_GSM8K_856 ergo_GSM8K_288 ergo_GSM8K_1047 ergo_GSM8K_799 ergo_GSM8K_932 \
    ergo_GSM8K_189 ergo_GSM8K_740 ergo_GSM8K_808 ergo_GSM8K_435 ergo_GSM8K_1247 ergo_GSM8K_953 \
    ergo_GSM8K_427 ergo_GSM8K_1246 ergo_GSM8K_143 ergo_GSM8K_359 ergo_GSM8K_144 ergo_GSM8K_576 \
    ergo_GSM8K_1027 ergo_GSM8K_698 ergo_GSM8K_1066 ergo_GSM8K_158 ergo_GSM8K_535 ergo_GSM8K_1059 \
    ergo_GSM8K_1166 ergo_GSM8K_929 ergo_GSM8K_1180 ergo_GSM8K_201 ergo_GSM8K_248 ergo_GSM8K_642 \
    ergo_GSM8K_598 ergo_GSM8K_1019 ergo_GSM8K_1269 ergo_GSM8K_584 ergo_GSM8K_465 ergo_GSM8K_991 \
    ergo_GSM8K_1197 ergo_GSM8K_1058 ergo_GSM8K_651 ergo_GSM8K_562 ergo_GSM8K_751 ergo_GSM8K_401 \
    ergo_GSM8K_916 ergo_GSM8K_1287 ergo_GSM8K_543 ergo_GSM8K_234 ergo_GSM8K_237 ergo_GSM8K_477 \
    ergo_GSM8K_534 ergo_GSM8K_214 ergo_GSM8K_547 ergo_GSM8K_215 ergo_GSM8K_122 ergo_GSM8K_2 \
    ergo_GSM8K_829 \
  --controller constant_remind
