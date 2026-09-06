#!/bin/bash
#SBATCH --job-name pdc-ergo-authority60-zero
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergo-authority60-zero-%j.out

# Sample-expansion follow-up to the 20-item minimal authority check (job
# 15602880/15602881, docs/experiments/ergo_multiturn_reliability_pilot.md):
# that pilot already confirmed the reset actuator has large, significant
# authority (paired t-test on final-turn task success, t=4.79, p=0.000127),
# so this isn't re-testing "does it work" -- it's firming up the effect
# size estimate and giving analysis_ergo_math.py's new-Q1/new-Q3 diagnostics
# more power, same reasoning as the sycophancy line's 20->60 item
# expansion. A FRESH 60-item draw (--item-rng-seed 1, not the pilot's 20
# items) -- the item pool only has 103 items total, so this and the
# sibling *_reset_60items.sbatch's shared --item-ids list intentionally
# overlaps some of the pilot's 20 items but isn't restricted to them.
# 60 items x 2 seeds = 684 total (item, turn) rows, no judge model calls
# (ergo_math_judge.py is pure regex) -- roughly 3x the pilot's 222 rows,
# --time scaled up accordingly from the pilot's ~6-14 minute actual runtime.

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
  --output-dir outputs/ergo_math_authority_zero_control_60items \
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
  --controller zero_control
