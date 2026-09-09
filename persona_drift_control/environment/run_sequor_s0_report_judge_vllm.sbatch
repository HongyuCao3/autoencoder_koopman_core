#!/bin/bash
#SBATCH --job-name pdc-sequor-s0-report-judge
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 02:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/sequor-s0-report-judge-%j.out

# S0 JUDGE CALIBRATION, REPORTING JUDGE CANDIDATE (instrument).
# docs/experiments/constraint_retention_plan.md section 3, gates G-S0-1 /
# G-S0-2; docs/experiments/constraint_signal_screening.md section 9.1, gate
# G-S0-0. Full 2000-row gold set, exactly label-balanced. No --limit, so the
# report carries mode="canonical" and its numbers ARE evidence.
#
# DO NOT SUBMIT BEFORE the backend consistency check
# (run_sequor_judge_vllm_consistency.sbatch) has passed its >= 22/24
# acceptance. That check is what makes a vLLM number comparable to the HF
# smoke run and to upstream's own calibration.
#
# WHAT IT DECIDES: the reporting judge for the whole `constraint` line -- the
# independent scorer every reported number must come from
# (.claude/global.md -> 报告口径). G-S0-1 (agreement >= 0.80 AND kappa >= 0.60)
# passes -> this is the reporting judge, S1 proceeds. G-S0-2 only
# (balanced >= 0.75 AND kappa >= 0.40) -> usable, but every number scored with
# it is annotated "judge weak -> MDE raised". Neither -> swap candidate ONCE
# (Qwen3-30B-A3B); a second failure closes the line (plan section 7, death
# condition 1).
#
# G-S0-0 is reported alongside, and it is the reason a bare accuracy is not
# enough: the gold set has a length confound (`violate` responses are
# systematically longer, median 478 vs 378 words), so a rule that reads no
# constraint at all scores 64.8% paired / 59.7% unpaired. The judge's margin
# over the PAIRED floor is what says it is reading the constraint at all.
#
# RUNTIME ESTIMATE 15-30 min central, ~1 h pessimistic. Assumptions:
# A100-PCIE-40GB; Qwen3-14B bf16 (28 GB weights, leaving ~6 GB of KV cache at
# --gpu-memory-utilization 0.90, i.e. ~40k tokens at ~160 KB/token for 40
# layers x 8 KV heads x 128 dim -> roughly 35 concurrent sequences);
# 2000 prompts averaging ~650 tokens (gold responses: median 429 words, mean
# 427, p90 679, max 1936) = ~1.3M prefill tokens at ~2.5-3.5k tok/s = 6-9 min;
# ~400k output tokens (median 185/row measured on the HF smoke) at
# ~0.8-1.9k tok/s = 4-8 min; model load + graph capture 3-8 min.
# --time 02:00:00 is >= 4x the central estimate. Reference point: the HF
# transformers path is 4.12 h for this same run (measured, job 15719117).
#
# No pip install: this env has vLLM and nothing else (see the consistency
# sbatch for why that is deliberate).

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/sequor_vllm

export HF_HOME=/scratch/hcao2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

nvidia-smi -L || true

python scripts/calibrate_sequor_judge_vllm.py \
  --judge-model Qwen/Qwen3-14B \
  --max-new-tokens 512 \
  --compare-to outputs/sequor_s0_smoke/judge_calibration_qwen3_14b_smoke.json \
  --out-path outputs/sequor_s0_report_judge_qwen3_14b/judge_calibration_qwen3_14b_vllm.json
