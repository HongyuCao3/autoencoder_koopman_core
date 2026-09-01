#!/bin/bash
#SBATCH --job-name pdc-adv-screening-thinking
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 03:00:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/adv-screening-thinking-%j.out

# enable_thinking=True rerun of run_adversarial_screening.sbatch's baseline
# (job 15399715, new-Q1/new-Q3 both passed at p<0.0001 in non-thinking mode)
# -- see docs/experiments/adversarial_screening_thinking_pilot.md for why.
# Same attack bank / seeds / num_attacks as the baseline (generation=thinking
# only changes ChatModel's enable_thinking flag and agent_max_new_tokens,
# see conf/generation/thinking.yaml), so the two runs' new-Q1/new-Q3 numbers
# are directly comparable.
#
# --time is 3x the non-thinking baseline's real 31 minutes, not a measured
# estimate for thinking mode -- reasoning traces mean both longer
# generations per turn (agent_max_new_tokens 256->1536) and no prior timing
# data at this token budget. If this gets killed by --time, DO NOT just
# resubmit this script: Hydra's hydra.run.dir uses ${now:...}, so a bare
# resubmit creates a *new* timestamped output dir and "resumes" an empty
# one. Instead find the killed run's directory under
# outputs/adversarial_screening_thinking_ablation/thinking/ and resubmit
# with that directory pinned explicitly:
#   sbatch environment/run_adversarial_screening_thinking.sbatch \
#     outputs/adversarial_screening_thinking_ablation/thinking/<that_timestamp>
# (the script below forwards $1 as a hydra.run.dir= override when given).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python -m pip install -e . -q

RESUME_DIR_OVERRIDE=()
if [[ "${1:-}" != "" ]]; then
  RESUME_DIR_OVERRIDE=("hydra.run.dir=${1}")
fi

python scripts/run_adversarial_screening_hydra.py generation=thinking "${RESUME_DIR_OVERRIDE[@]}"
