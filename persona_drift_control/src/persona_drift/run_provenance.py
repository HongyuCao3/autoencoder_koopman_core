"""Harness fingerprint for run artifacts.

`.claude/global.md` -> *产物与谱系* makes this a hard constraint: every run
directory must be able to answer WHICH HARNESS VERSION PRODUCED IT. Without
that field an instrument fix cannot be scoped, so it forces a full re-run of
everything -- the direct cause of the 18 ERGO jobs that re-answered one
question (`docs/LEDGER.md` section 2). No artifact in this repo carried the
field before 2026-09-09, including `outputs/ergo_ekA_branch/run_config.json`,
which is why this module exists rather than another per-script dict literal.

It **refuses rather than degrades**: a record with `git_sha: null` looks like
provenance in a JSON file while answering nothing, and the failure mode this
field exists to prevent is invisible in the artifact. Same reasoning as
`run_config_guard` -- a guard that silently passes protects nothing.

Deliberately free of torch/pandas/transformers imports: the `constraint`
line's vLLM environment (`docs/experiments/constraint_retention_plan.md`
section 8) has vLLM and nothing else, and it writes artifacts too.
"""

from __future__ import annotations

import os
import pathlib
import platform
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_STATUS_CHARS = 4000


class MissingProvenance(RuntimeError):
    """Raised when the harness version cannot be determined."""


def _git(repo: pathlib.Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MissingProvenance(f"cannot run git in {repo}: {exc}") from exc
    if proc.returncode != 0:
        raise MissingProvenance(
            f"`git {' '.join(args)}` failed in {repo} (exit {proc.returncode}): "
            f"{proc.stderr.strip()}. An artifact without a harness fingerprint "
            f"cannot be re-run selectively -- see .claude/global.md 产物与谱系."
        )
    return proc.stdout.strip()


def provenance(switches: dict | None = None, repo: pathlib.Path | None = None) -> dict:
    """The harness fingerprint to embed in a run artifact.

    `switches` is the run's own key/value settings (judge model, backend,
    decoding caps, seed): global.md asks for the sha **and** the switches,
    because two runs at the same sha with different switches are different
    instruments.

    `git_dirty` is reported, not refused: the honest state of a session that
    is writing the code it runs. A canonical run should still be launched from
    a committed tree, or the sha names something other than what ran.
    """

    repo = repo or REPO_ROOT
    status = _git(repo, "status", "--porcelain")
    dirty_paths = [line for line in status.splitlines() if line.strip()]
    return {
        "git_sha": _git(repo, "rev-parse", "HEAD"),
        "git_branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty_paths),
        "n_dirty_paths": len(dirty_paths),
        "git_status_short": status[:_STATUS_CHARS] or None,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": platform.node(),
        "python": sys.version.split()[0],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "switches": dict(switches or {}),
    }
