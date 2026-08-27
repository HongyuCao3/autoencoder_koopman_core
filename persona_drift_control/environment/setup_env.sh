#!/usr/bin/env bash
# Run this in a real interactive shell on the cluster (module/conda are not
# available from the sandboxed shell used to write this repo). Not meant to
# be sourced blindly - read it, adjust the anaconda3 module version, then
# run the commands, e.g. `bash -x environment/setup_env.sh` or paste by hand.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX=/scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

# Adjust to whatever `module avail anaconda3` / `module avail miniconda3`
# reports on this cluster.
module load anaconda3

conda create -y -p "$ENV_PREFIX" python=3.10
source activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT[test]"

mkdir -p "$NLTK_DATA"
python -m nltk.downloader -d "$NLTK_DATA" vader_lexicon punkt punkt_tab \
  averaged_perceptron_tagger averaged_perceptron_tagger_eng || true

cat <<EOF

Environment ready at: $ENV_PREFIX

For every future job/session that runs this code, set (do this BEFORE
importing transformers/nltk, e.g. at the top of an sbatch script, so nothing
ever gets written under /home or the node-local disk by accident):

  export HF_HOME=/scratch/hcao2/hf_cache
  export NLTK_DATA=/scratch/hcao2/nltk_data
  source activate $ENV_PREFIX

The Qwen3-4B / Qwen3-8B weights are NOT downloaded yet - the first run of
scripts/run_signal_screening.py will pull them from the Hugging Face Hub into
HF_HOME above (several GB). If compute nodes on this cluster lack internet
access, pre-warm the cache from the login node first, e.g.:

  python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \\
    AutoTokenizer.from_pretrained('Qwen/Qwen3-4B'); \\
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')"
EOF
