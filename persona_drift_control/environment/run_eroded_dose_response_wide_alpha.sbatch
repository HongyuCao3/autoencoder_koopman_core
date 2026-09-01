#!/bin/bash
#SBATCH --job-name pdc-dose-response-wide-alpha
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/dose-response-wide-alpha-%j.out

# Candidate 1 from docs/experiments/dose_response_pilot.md "下一步" (2026-08-31): both prior
# dose-response attempts (jobs 15404914, 15405662) tested alpha in {-1,-0.5,0,0.5,1} and found
# no significant response. This widens the grid to {-5,-3,-2,-1,-0.5,0,0.5,1,2,3,5} on the same
# eroded-context query set (job 15405662's design) to check whether the direction is right but
# too weak at the original scale, or genuinely has no causal effect at this layer/application
# point. Separate --output-dir from job 15405662's so that result stays available for comparison.
# 19 queries x 11 alphas = 209 generations, ~2x job 15405662's cost -- --time is a generous
# upper bound based on that job's ~8s/generation average. Not resumable (same reasoning as the
# other dose-response jobs) -- a --time kill just needs resubmission.

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
  --output-dir outputs/dose_response_eroded_wide_alpha \
  --alpha-grid -5 -3 -2 -1 -0.5 0 0.5 1 2 3 5
