#!/bin/bash
#SBATCH --job-name pdc-v-align-fix
#SBATCH --cpus-per-task 2
#SBATCH --mem 4G
#SBATCH --time 00:15:00
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/v-align-fix-%j.out

# docs/next step.md 第一步（零 GPU）: 验证 "v 槽位错位" 诊断
# (docs/experiments/koopman_defense_pilot.md, 2026-09-02 追加分析) 的两条
# 可证伪预测。CPU-only -- 只是没有直接在写代码的沙箱里跑 python 的能力
# (RUNNING_ON_PALMETTO.md 记录过这个沙箱限制), 所以借用现成的 sbatch 通道。
#
# 先跑改动过的两个测试文件 (dataset.py/control.py 新增的
# contemporaneous_v 对齐开关), 确认默认行为 (shift=0) 完全不变、新分支的
# 手算用例通过; 再跑 scripts/analyze_v_alignment_fix.py。

set -euo pipefail

module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data

cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

python -m pip install -e ".[test]" -q

echo "=== pytest (modified files) ==="
pytest -q tests/test_modeling_dataset.py tests/test_control.py

echo
echo "=== full CPU test suite (regression guard) ==="
pytest -q

echo
echo "=== analyze_v_alignment_fix.py ==="
python scripts/analyze_v_alignment_fix.py

echo
echo "=== analyze_phaseB_seed_confound.py ==="
python scripts/analyze_phaseB_seed_confound.py

echo
echo "=== analyze_compounding_hypothesis.py ==="
python scripts/analyze_compounding_hypothesis.py

echo
echo "=== analyze_mu2_interaction_replay.py ==="
python scripts/analyze_mu2_interaction_replay.py

echo
echo "=== analyze_state_action_interaction.py --contemporaneous-v (fit + save the model for the real closed-loop validation) ==="
python scripts/analyze_state_action_interaction.py --contemporaneous-v \
  --out-path outputs/koopman_case_study/interaction_model_report_valigned.json

echo
echo "=== analyze_mu2_interaction_replay.py --pad-short-history (against Phase I's own real trajectories) ==="
python scripts/analyze_mu2_interaction_replay.py --pad-short-history \
  --replay-path outputs/koopman_defense_phaseI_koopman_mpc_valigned/trajectories.jsonl \
  --out-path outputs/koopman_case_study/mu2_interaction_replay_padded_report.json
