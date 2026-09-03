#!/bin/bash
#SBATCH --job-name pdc-koopman-phaseG-periodic
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/koopman-phaseG-periodic-%j.out

# Phase G, periodic arm (docs/evaluation/BASELINES.md's "周期性/事件触发重提醒" candidate,
# implemented as control.py::PeriodicController -- fixed-schedule reminder
# every --periodic-period turns, no feedback signal at all). period=2 chosen
# to land close to koopman_mpc's empirical Phase E insertion rate (32/80 =
# 40%; period=2 gives ~40% on these 4-6 turn attack trajectories), so the
# comparison against koopman_mpc is at roughly matched cost. Same 8 held-out
# attacks as every other Phase E/G arm -- see
# run_koopman_defense_phaseE_zero_control.sbatch.

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
  --output-dir outputs/koopman_defense_phaseG_periodic \
  --seeds 0 1 \
  --controller periodic --periodic-period 2 \
  --attack-ids safemtdata_0074 safemtdata_0169 safemtdata_0257 safemtdata_0289 \
    safemtdata_0324 safemtdata_0329 safemtdata_0476 safemtdata_0530
