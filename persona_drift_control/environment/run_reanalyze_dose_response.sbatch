#!/bin/bash
#SBATCH --job-name pdc-reanalyze-dose-response
#SBATCH --cpus-per-task 2
#SBATCH --mem 4G
#SBATCH --time 00:05:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/reanalyze-dose-response-%j.out

# Ad hoc, CPU-only: re-runs new-Q2 on job 15406762's wide-alpha data
# restricted to different alpha subsets, to check whether the |alpha|=5
# degenerate-text points are driving the significant result (see
# docs/experiments/dose_response_pilot.md).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

python -m pip install -e . -q

echo "=== full grid (includes |alpha|=5 degenerate points) ==="
python scripts/reanalyze_dose_response_subset.py --rows-path outputs/dose_response_eroded_wide_alpha/dose_response_rows.jsonl

echo
echo "=== excluding |alpha|=5 ==="
python scripts/reanalyze_dose_response_subset.py --rows-path outputs/dose_response_eroded_wide_alpha/dose_response_rows.jsonl \
  --alpha-grid -3 -2 -1 -0.5 0 0.5 1 2 3

echo
echo "=== non-negative arm only (0 to 3, the cleanest monotonic segment) ==="
python scripts/reanalyze_dose_response_subset.py --rows-path outputs/dose_response_eroded_wide_alpha/dose_response_rows.jsonl \
  --alpha-grid 0 0.5 1 2 3

echo
echo "=== non-positive arm only (-3 to 0) ==="
python scripts/reanalyze_dose_response_subset.py --rows-path outputs/dose_response_eroded_wide_alpha/dose_response_rows.jsonl \
  --alpha-grid -3 -2 -1 -0.5 0
