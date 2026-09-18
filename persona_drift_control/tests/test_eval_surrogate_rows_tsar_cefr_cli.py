"""The CLI boundary of `eval_surrogate_rows_tsar_cefr.py`.

`run()` records the rows path relative to the repo root as its LAST act, after
the whole five-fold fit. A relative `--rows` therefore used to raise there with
every number already computed and nothing on disk -- a 75-minute gemma fit died
that way on 2026-09-17. Resolving at the boundary is the fix; this test is what
stops it coming back, because the failure is invisible until the end of a run
too long to notice it in.
"""
import importlib.util
import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "eval_surrogate_rows_tsar_cefr_cli", SCRIPTS / "eval_surrogate_rows_tsar_cefr.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

STUB = {"lag": 1, "horizon": 4, "n_items": 0, "n_trajectories": 0, "n_scored_rows": 0,
        "best_null_name": "stateless", "rows": {}, "contrasts": {}}


@pytest.fixture
def fake_run(monkeypatch):
    """Stands in for the fit, but keeps the one line that used to raise."""
    def _run(rows_path, bootstrap_seed=mod.BOOTSTRAP_SEED):
        return dict(STUB, rows_path=str(rows_path.relative_to(mod.REPO_ROOT)),
                    n_trajectories_refused_unparsed_readout=0)
    monkeypatch.setattr(mod, "run", _run)


def test_relative_rows_path_survives_the_provenance_line(tmp_path, monkeypatch, fake_run):
    rows = mod.PDC_ROOT / "outputs" / "tsar_cefr_gpu1" / "trajectories.jsonl"
    relative = rows.relative_to(pathlib.Path.cwd()) if str(rows).startswith(str(pathlib.Path.cwd())) \
        else pathlib.Path("persona_drift_control/outputs/tsar_cefr_gpu1/trajectories.jsonl")
    monkeypatch.chdir(mod.REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["prog", "--rows", str(relative),
                                      "--out-dir", str(tmp_path / "out")])
    mod.main()
    written = (tmp_path / "out" / "tsar_cefr.json").read_text()
    assert "tsar_cefr_gpu1/trajectories.jsonl" in written


def test_out_dir_is_resolved_too(tmp_path, monkeypatch, fake_run):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "landing").mkdir()
    monkeypatch.setattr(sys, "argv", ["prog", "--out-dir", "landing"])
    mod.main()
    assert (tmp_path / "landing" / "tsar_cefr.json").exists()


def test_existing_output_is_still_refused(tmp_path, monkeypatch, fake_run):
    out = tmp_path / "out"
    out.mkdir()
    (out / "tsar_cefr.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["prog", "--out-dir", str(out)])
    with pytest.raises(FileExistsError):
        mod.main()
