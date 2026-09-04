#!/bin/bash
#SBATCH --job-name pdc-lstm-valigned
#SBATCH --cpus-per-task 4
#SBATCH --mem 8G
#SBATCH --time 00:30:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/lstm-valigned-%j.out

# Re-runs the LSTM surrogate baseline under the CORRECTED v alignment
# (docs/experiments/lstm_baseline_plan.md's "待复核" note): the 2026-09-01
# numbers in that doc were produced before modeling/dataset.py's v-alignment
# fix, so they are only comparable to koopman_fit_report.json's
# richer_abs_sign (0.0430), not to the v-aligned 0.0684 that the AE baseline
# and everything since is scored against. This closes the one baseline still
# reported on the pre-fix footing.
#
# Two deliberate differences from the original run, both recorded in the plan
# doc's follow-up section:
#   * --train-seeds 0 1 2 instead of a single seed -- the AE baseline found a
#     one-seed result that reversed with two more seeds.
#   * --min-turn-index is derived, not hard-coded: with contemporaneous_v the
#     Koopman window starts at max(nu-1, mu-1) = 1, not 2, so scoring the
#     LSTM from turn 2 would compare the two models on different turn sets.
#
# CPU-only. Runs through sbatch only because the code-writing sandbox has no
# python (RUNNING_ON_PALMETTO.md), not because it needs a node.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

python -m pip install -e ".[test]" -q

echo "=== pytest (modified file) ==="
pytest -q tests/test_lstm_baseline.py

echo
echo "=== full CPU test suite (regression guard) ==="
pytest -q

echo
echo "=== fit_koopman_lstm_baseline.py (v-aligned, 3 train seeds) ==="
python -u scripts/fit_koopman_lstm_baseline.py

echo
echo "=== control: same script, pre-fix alignment + single seed (must reproduce the 2026-09-01 table) ==="
python -u scripts/fit_koopman_lstm_baseline.py \
  --no-contemporaneous-v \
  --train-seeds 0 \
  --koopman-fit-report outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json \
  --out-path outputs/koopman_lstm_baseline/lstm_fit_report_prefix_reproduction.json

echo
echo "=== fair-protocol variant: early-stop on a validation split of the TRAINING attacks ==="
# The 2026-09-01 protocol early-stops on the same held-out trajectories it
# then reports, so the reported MSE is a best-of-300-epochs number selected
# BY that set. That was harmless while the LSTM lost by ~2x; it is not
# harmless now that the v-aligned run lands on top of the linear baseline,
# because the optimistic side is exactly the side that closed the gap.
# val_frac=0.15 matches the AE baseline's own early-stopping split, and the
# held-out attack set is asserted to be unchanged (see the script).
python -u scripts/fit_koopman_lstm_baseline.py \
  --val-frac 0.15 \
  --out-path outputs/koopman_lstm_baseline/lstm_fit_report_valigned_valsplit.json

echo
echo "=== confound control: same reduced training set, but early-stop on held-out again ==="
# Carving out the validation split also costs the LSTM 4 of its 22 training
# attacks, and that alone would raise the MSE. This arm holds the training
# set at the reduced 18 attacks and puts early stopping back on the reported
# trajectories, so the two effects can be read off separately instead of
# being attributed wholesale to the fair protocol.
python -u scripts/fit_koopman_lstm_baseline.py \
  --val-frac 0.15 \
  --early-stop-on held_out \
  --out-path outputs/koopman_lstm_baseline/lstm_fit_report_valigned_lesstrain_earlystop_heldout.json
