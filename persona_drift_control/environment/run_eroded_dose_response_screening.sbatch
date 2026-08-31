#!/bin/bash
#SBATCH --job-name pdc-dose-response-eroded
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/dose-response-eroded-%j.out

# Eroded-context dose-response variant (docs/experiments/dose_response_pilot.md
# "下一步" candidate 1): the first attempt (run_dose_response_screening.sbatch,
# job 15404914) asked the attack's raw harmful goal as a bare first turn and
# hit a ceiling effect (near-perfect refusal at every alpha). This instead
# steers only the final turn of step-1 screening's REAL recorded
# conversations (outputs/adversarial_screening/trajectories.jsonl, job
# 15399715), so the steered generation starts from wherever safety had
# actually eroded to. Requires both that file and
# outputs/safety_direction/safety_direction.npy (run_calibrate_safety_
# direction.sbatch) to already exist -- --layer here must match calibration.
# ~20 queries x 5 alpha = ~100 generations, same order of cost as the first
# dose-response job. Resumable via dose_response_rows.jsonl -- a --time kill
# just needs `sbatch` on this same script again.

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q
python scripts/run_eroded_dose_response_screening.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --layer 18 \
  --direction-path outputs/safety_direction/safety_direction.npy \
  --trajectories-path outputs/adversarial_screening/trajectories.jsonl \
  --output-dir outputs/dose_response_eroded
