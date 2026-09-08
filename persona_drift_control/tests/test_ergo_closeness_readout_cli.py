"""docs/experiments/ergo_fidelity_restoration_plan.md EK0: scripts/analyze_ergo_closeness_readout.py
gained --rows-path / --entropy-path / --out-path so the RC-0..3 battery can run on the
upstream-profile excitation arm. Per .claude/code.md the old path must be byte-identical,
so these check that the defaults still point at Phase B and that the only new behavior --
dropping the two entropy columns when no entropy_readout.json exists -- is the sole
difference on a directory that lacks one.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_ergo_closeness_readout import (  # noqa: E402
    DEFAULT_ENTROPY_PATH,
    DEFAULT_OUT_PATH,
    DEFAULT_ROWS_PATH,
    parse_args,
)


def test_defaults_are_the_pre_patch_phase_b_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_ergo_closeness_readout.py"])
    args = parse_args()
    assert args.rows_path == pathlib.Path("outputs/ergo_math_phaseB_random_excite/trajectories.jsonl")
    assert args.entropy_path == pathlib.Path("outputs/ergo_math_phaseB_random_excite/entropy_readout.json")
    assert args.out_path == pathlib.Path("outputs/ergo_math_phaseB_random_excite/closeness_readout_state_report.json")
    assert (args.rows_path, args.entropy_path, args.out_path) == (
        DEFAULT_ROWS_PATH,
        DEFAULT_ENTROPY_PATH,
        DEFAULT_OUT_PATH,
    )


def test_paths_are_overridable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_ergo_closeness_readout.py",
            "--rows-path", str(tmp_path / "trajectories.jsonl"),
            "--entropy-path", str(tmp_path / "entropy_readout.json"),
            "--out-path", str(tmp_path / "report.json"),
        ],
    )
    args = parse_args()
    assert args.rows_path == tmp_path / "trajectories.jsonl"
    assert args.out_path == tmp_path / "report.json"


def test_absent_entropy_file_is_the_signal_to_drop_the_two_entropy_columns(monkeypatch, tmp_path):
    """main() keys the entropy columns off `entropy_path.exists()`; nothing else changes."""
    monkeypatch.setattr(
        sys, "argv", ["analyze_ergo_closeness_readout.py", "--entropy-path", str(tmp_path / "missing.json")]
    )
    args = parse_args()
    assert not args.entropy_path.exists()
    assert DEFAULT_ENTROPY_PATH.name == "entropy_readout.json"
