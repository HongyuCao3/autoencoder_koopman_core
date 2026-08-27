# Running on Palmetto 2 (Slurm)

This cluster is Clemson RCD's Palmetto 2, scheduled by Slurm. Full docs:
https://docs.rcd.clemson.edu/palmetto/. This file only covers what's needed
to debug and run `scripts/run_signal_screening.py`.

**Update 2026-08-27: validated end to end.** The write-code sandbox this
project was authored in has `sbatch`/`squeue` (they reach the real Slurm
controller) but no working `module`/`conda`/`python`/`scontrol`/`srun --pty`
(missing shared libs in that specific container - `sacctmgr` and `scontrol`
fail with `libhistory.so.7` errors, `srun` fails loading MPI/http_parser
plugins; none of that reflects your own login/OnDemand shell, which is a
normal cluster environment). From there we still submitted two real jobs:
- a trivial 1-CPU probe job, to confirm `sbatch` actually reaches the
  scheduler and a job comes back with real output;
- `environment/run_smoke_test.sbatch` (checked into this repo), which
  loaded `anaconda3`, activated `/scratch/hcao2/envs/persona_drift_pilot`,
  got an A100 80GB (`--gpus a100:1` on the default `work1` partition, no
  `--account` needed), downloaded Qwen3-4B into
  `/scratch/hcao2/hf_cache`, and ran a real (tiny) self-chat + probe-scoring
  pass through `run_signal_screening.py` - `trajectories.jsonl` and
  `screening_report.md` came out with the expected schema. `overall_pass`
  was `False`, which is expected and uninformative at that scale (1 prompt,
  2 turns, 1 probe repeat - not the real gate questions), not a signal
  about the real run.

One finding from reading that smoke-test output, now fixed:
`selfchat._looks_like_refusal`'s `"as an ai"` marker fired on a response
that just used the phrase while still answering normally ("As an AI, I
don't have real-time access to current events, but...") - a genuine
refusal-heuristic false positive, not a pipeline bug. The marker has been
dropped; re-run the smoke test if you want to confirm `refusal_rate` looks
more sane before trusting it on the real screening run.

## Before anything: find your account/partition

```bash
sacctmgr show associations user=$USER format=account,partition,qos
sinfo -s
```

Most users default to the general `work1` partition with no `--account`
needed (confirmed above); if you have a purchased/owner partition, prefer it
for priority and the longer wall-time limit. GPU type codes: `h200 h100
l40s a40 a100 v100s v100 p100` (colleague's README used A100/4090 for
Qwen3-4B; `a100` worked fine above and is a safe default, adjust if your
allocation doesn't have it).

## 1. Interactive debug (do this first, not a batch job)

Get a GPU shell and run the pilot directly so you see errors immediately
instead of waiting on a queued batch job + log file. (This `salloc` path
itself wasn't testable from the sandbox above - only `sbatch` reaches the
scheduler there - but `environment/run_smoke_test.sbatch` exercises the same
commands as a batch job and is confirmed working; use whichever fits, `sbatch
environment/run_smoke_test.sbatch` is the fastest way to re-check the pilot
still runs.)

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

`environment/run_smoke_test.sbatch` is the validated template (loads
`anaconda3`, activates the env, sets `HF_HOME`/`NLTK_DATA`, requests
`a100:1`) - copy it for the real screening run rather than writing one from
scratch:

```bash
cp environment/run_smoke_test.sbatch environment/run_screening.sbatch
# edit run_screening.sbatch:
#   --job-name persona-drift-screening
#   --time 03:00:00   (see note below)
#   --output .../environment/slurm_logs/screening-%j.out
#   drop --num-prompts/--seeds/--num-turns/--probe-repeats to use the
#   protocol-scale defaults (5 prompts, seeds 0 1, 16 turns, 4 probe repeats)
#   --output-dir outputs/signal_screening

sbatch environment/run_screening.sbatch
```

`--time 03:00:00` is a guess based on README's "~1 hour for the protocol's
literal 10 trajectories" doubled for the 20 trajectories `screening.py`
actually runs, plus slack for model download on first run (the smoke test's
Qwen3-4B download only took ~20s, so that slack is generous) - watch the
first real run and tighten this once you know the actual wall-clock.

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
