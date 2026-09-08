#!/bin/bash
#SBATCH --job-name pdc-ergoEKA-branch
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 04:00:00
#SBATCH --requeue
#SBATCH --gpus a100:1
#SBATCH --output /home/hcao2/autoencoder_koopman_core/persona_drift_control/environment/slurm_logs/ergoEKA-branch-%j.out

# EK-A (docs/experiments/ergo_fidelity_restoration_plan.md section 4): the
# counterfactual-branch identification arm. Copied from
# environment/run_ergo_upB_random_excite.sbatch except job-name, log path,
# --requeue, --time, the flock, and the command itself.
#
# --item-ids is the 45 items Phase B uses that the held-out evaluation set does
# NOT (the 103-item bank splits 45 / 15 shared / 43). Spelled out rather than
# derived at run time so the arm's item set is auditable from this file and
# cannot drift when the bank or a selection seed changes. EK-B evaluates on the
# 58 held-out items, so identification and evaluation stay disjoint.
#
# --time 04:00:00, estimated not copied (failure mode 14): 45 items x 3 seeds =
# 135 trajectories; each generates every turn TWICE (base + counterfactual), and
# R1's upstream-profile arms measured 19-23 s per single-generation trajectory,
# so ~42 s each -> ~95 min. 04:00:00 is ~2.5x headroom.
#
# --requeue: this queue preempts (K1/15696221 was preempted and rescheduled
# under the same job id). Safe here because the runner resumes per base
# trajectory_id and only counts a trajectory done when BOTH output files
# contain it.
#
# The pip install is serialized with flock: two jobs installing this package
# into the same shared env concurrently is what killed 15696222 and 15696227 on
# 2026-09-08 (see .claude/experiments.md -> before submitting a GPU job).

set -euo pipefail
module load anaconda3
source activate /scratch/hcao2/envs/persona_drift_pilot
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control
nvidia-smi -L || true
mkdir -p /scratch/hcao2/envs/.locks
flock /scratch/hcao2/envs/.locks/persona_drift_pilot.editable.lock python -m pip install -e . -q

python scripts/run_ergo_branch_arm.py \
  --agent-model Qwen/Qwen3-4B \
  --device cuda \
  --output-dir outputs/ergo_ekA_branch \
  --base-controller zero_control \
  --seeds 0 1 2 \
  --reset-mode append \
  --prompt-profile upstream \
  --item-ids ergo_GSM8K_1011 ergo_GSM8K_1019 ergo_GSM8K_1058 ergo_GSM8K_1059 ergo_GSM8K_1113 ergo_GSM8K_1124 ergo_GSM8K_117 ergo_GSM8K_1180 ergo_GSM8K_1187 ergo_GSM8K_1190 ergo_GSM8K_1197 ergo_GSM8K_1207 ergo_GSM8K_1240 ergo_GSM8K_1287 ergo_GSM8K_140 ergo_GSM8K_144 ergo_GSM8K_2 ergo_GSM8K_201 ergo_GSM8K_215 ergo_GSM8K_359 ergo_GSM8K_378 ergo_GSM8K_417 ergo_GSM8K_44 ergo_GSM8K_477 ergo_GSM8K_535 ergo_GSM8K_543 ergo_GSM8K_562 ergo_GSM8K_584 ergo_GSM8K_598 ergo_GSM8K_621 ergo_GSM8K_642 ergo_GSM8K_651 ergo_GSM8K_698 ergo_GSM8K_739 ergo_GSM8K_752 ergo_GSM8K_784 ergo_GSM8K_825 ergo_GSM8K_871 ergo_GSM8K_875 ergo_GSM8K_913 ergo_GSM8K_927 ergo_GSM8K_941 ergo_GSM8K_953 ergo_GSM8K_961 ergo_GSM8K_976
