"""Tests for persona_drift.run_provenance.

The load-bearing test is `test_a_non_repo_raises_instead_of_returning_a_null_sha`:
the whole point of the module is that a missing harness fingerprint stops the
run. A version that returned `{"git_sha": None}` would put a provenance-shaped
dict in every artifact while answering nothing, and the failure it exists to
prevent (an instrument fix that cannot be scoped, so everything is re-run)
would be invisible in the artifacts -- the same argument the
`run_config_guard` meta-test makes.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from persona_drift.run_provenance import REPO_ROOT, MissingProvenance, provenance


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_git_sha_matches_the_repo_head():
    assert provenance()["git_sha"] == _git(REPO_ROOT, "rev-parse", "HEAD")


def test_sha_is_a_full_hex_sha():
    sha = provenance()["git_sha"]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_dirty_flag_agrees_with_git_status():
    prov = provenance()
    dirty = bool(_git(REPO_ROOT, "status", "--porcelain"))
    assert prov["git_dirty"] is dirty
    assert (prov["n_dirty_paths"] > 0) is dirty


def test_a_non_repo_raises_instead_of_returning_a_null_sha(tmp_path):
    with pytest.raises(MissingProvenance):
        provenance(repo=tmp_path)


def test_missing_git_binary_raises(monkeypatch, tmp_path):
    """A compute node without git on PATH must fail at second one, loudly,
    not four hours later with an untraceable artifact."""

    def no_git(*_args, **_kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", no_git)
    with pytest.raises(MissingProvenance):
        provenance()


def test_slurm_job_id_is_picked_up(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "15719117")
    monkeypatch.setenv("SLURM_JOB_NAME", "pdc-sequor-judge-smoke")
    prov = provenance()
    assert prov["job_id"] == "15719117"
    assert prov["job_name"] == "pdc-sequor-judge-smoke"


def test_job_id_is_none_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert provenance()["job_id"] is None


def test_switches_are_carried_verbatim():
    switches = {"judge_model": "Qwen/Qwen3-14B", "backend": "vllm", "max_new_tokens": 512}
    assert provenance(switches=switches)["switches"] == switches


def test_no_heavy_imports(monkeypatch):
    """The vLLM env (plan section 8) has vLLM and nothing else, and it writes
    artifacts too, so this module must not reach for the analysis stack."""

    import importlib
    import sys

    for blocked in ("torch", "pandas", "transformers"):
        monkeypatch.setitem(sys.modules, blocked, None)
    module = importlib.reload(importlib.import_module("persona_drift.run_provenance"))
    assert module.provenance()["git_sha"]
