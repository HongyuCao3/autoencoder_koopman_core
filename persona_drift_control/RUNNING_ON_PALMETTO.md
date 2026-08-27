# Running on Palmetto 2 (Slurm)

This cluster is Clemson RCD's Palmetto 2, scheduled by Slurm. Full docs:
https://docs.rcd.clemson.edu/palmetto/. This file only covers what's needed
to debug and run `scripts/run_signal_screening.py`.

The write-code sandbox this project was authored in has no `module`/`conda`/
`sbatch` in its shell, so none of the commands below have actually been run
end-to-end - sanity-check the resource numbers (especially `--mem` and GPU
type) against your own job's behavior the first time.

## Before anything: find your account/partition

```bash
sacctmgr show associations user=$USER format=account,partition,qos
sinfo -s
```

Most users default to the general `work1` partition with no `--account`
needed; if you have a purchased/owner partition, prefer it for priority and
the longer wall-time limit. GPU type codes: `h200 h100 l40s a40 a100 v100s
v100 p100` (colleague's README used A100/4090 for Qwen3-4B; `a100` is a safe
default here too, adjust if your allocation doesn't have it).

## 1. Interactive debug (do this first, not a batch job)

Get a GPU shell and run the pilot directly so you see errors immediately
instead of waiting on a queued batch job + log file:

```bash
salloc --cpus-per-task 8 --mem 32G --time 01:00:00 --gpus a100:1

# inside the allocation:
export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
source activate /scratch/hcao2/envs/persona_drift_pilot
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

# cheapest possible smoke test before the real screening run:
# 1 prompt, 1 seed, 2 turns, 1 probe repeat - just proves the pipeline runs
# and downloads/loads the models, in a couple of minutes instead of ~1-2h.
python scripts/run_signal_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B \
  --device cuda --num-prompts 1 --seeds 0 --num-turns 2 --probe-repeats 1 \
  --output-dir outputs/smoke_test
```

If that finishes and `outputs/smoke_test/screening_report.md` looks
sane, exit the allocation (`exit` or `Ctrl-D`) and run the real screening
(section 7 scale: `--num-prompts 5 --seeds 0 1 --num-turns 16
--probe-repeats 4`, the defaults) either interactively (bump `--time`) or as
a batch job below.

## 2. Batch job

```bash
#!/bin/bash
#SBATCH --job-name persona-drift-screening
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 03:00:00
#SBATCH --gpus a100:1
#SBATCH --output slurm-%j.out

export HF_HOME=/scratch/hcao2/hf_cache
export NLTK_DATA=/scratch/hcao2/nltk_data
source activate /scratch/hcao2/envs/persona_drift_pilot
cd /home/hcao2/autoencoder_koopman_core/persona_drift_control

srun python scripts/run_signal_screening.py \
  --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B --device cuda \
  --output-dir outputs/signal_screening
```

Save as e.g. `run_screening.sbatch`, then:

```bash
sbatch run_screening.sbatch
```

`--time 03:00:00` is a guess based on README's "~1 hour for the protocol's
literal 10 trajectories" doubled for the 20 trajectories `screening.py`
actually runs, plus slack for model download on first run - watch the first
real run and tighten this once you know the actual wall-clock.

## 3. Monitor / debug a running or queued job

```bash
squeue --me                              # is it running, pending, on which node
tail -f slurm-<jobid>.out                # live stdout/stderr
scontrol show job <jobid>                # full resource/state detail
scancel <jobid>                          # kill it
```

To get a live shell on the same node as a running job (e.g. to run
`nvidia-smi` or poke at a hung process):

```bash
squeue --me                              # read the NODELIST column
srun --jobid <jobid> --overlap --nodelist <hostname> --pty bash --login
```
