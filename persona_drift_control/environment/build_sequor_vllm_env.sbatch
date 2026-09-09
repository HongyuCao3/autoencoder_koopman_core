#!/bin/bash
#SBATCH --job-name pdc-build-sequor-vllm
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/build-sequor-vllm-%j.out

# Builds the `constraint` line's own vLLM environment
# (docs/experiments/constraint_retention_plan.md section 8). CPU only -- no
# --gpus line, this is an install.
#
# WHY A SEPARATE ENV: the shared /scratch/hcao2/envs/persona_drift_pilot is
# what every other line runs on, and vLLM pins its own torch. Installing it
# there would silently change the numerical stack under `defense`, `stance`
# and ERGO's already-published artifacts. It is also the env whose concurrent
# editable installs killed jobs 15696222 / 15696227 on 2026-09-08; nothing
# here touches it.
#
# WHY NOW: job 15719117 measured the judge at 7.42 s/row on HF transformers,
# so the full 2000-row S0 calibration is 4.12 h per candidate and S1's 9600
# judge calls would be ~20 h. The plan's own arithmetic (section 9) is ~8-12
# GPU-h for the whole line with vLLM against ~100 without. The >=3 s/row
# branch was written into the ledger before this ran.
#
# The env is a directory in /scratch; deleting it and re-running this job is
# the whole rollback procedure.

set -euo pipefail

module load anaconda3

ENV_PREFIX=/scratch/hcao2/envs/sequor_vllm

if [ -d "$ENV_PREFIX" ]; then
  echo "REFUSING: $ENV_PREFIX already exists. Delete it deliberately before rebuilding."
  exit 1
fi

conda create -y -p "$ENV_PREFIX" python=3.11
source activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install vllm

echo "=== versions ==="
python -c "import sys, torch, vllm; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__)"
python -c "import vllm; from vllm import LLM, SamplingParams; print('vllm entrypoints import OK')"
echo "=== env built at $ENV_PREFIX ==="
du -sh "$ENV_PREFIX"
