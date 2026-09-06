"""Guard against the E7 defect: a v-aligned interaction model reported against
old-alignment reference MSEs.

`analyze_state_action_interaction.py` copies the arx/richer_abs_sign held-out
MSEs verbatim out of `--koopman-fit-report` instead of recomputing them, so the
run is only self-consistent while that report shares this run's
`contemporaneous_v` (and nu/mu). It did not once -- see
paper/evidence/superseded.md, event E7.
"""

import argparse
import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_state_action_interaction.py"
_spec = importlib.util.spec_from_file_location("_asai", _SCRIPT)
_asai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_asai)


def _args(contemporaneous_v: bool, nu: int = 1, mu: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        contemporaneous_v=contemporaneous_v,
        nu=nu,
        mu=mu,
        koopman_fit_report=pathlib.Path("outputs/whatever/koopman_fit_report.json"),
    )


def _report(config: dict) -> dict:
    return {"config": config, "arx": {"held_out_rollout_mse": 0.0}, "richer_abs_sign": {"held_out_rollout_mse": 0.0}}


def test_matched_alignment_passes_for_both_settings():
    _asai._assert_reference_alignment_matches(_report({"nu": 1, "mu": 2, "contemporaneous_v": True}), _args(True))
    _asai._assert_reference_alignment_matches(_report({"nu": 1, "mu": 2, "contemporaneous_v": False}), _args(False))


def test_the_actual_e7_pairing_is_refused():
    # v-aligned run pointed at the pre-fix report: exactly what produced
    # interaction_model_report_valigned.json's 0.0510 / 0.0430 references.
    with pytest.raises(SystemExit, match="contemporaneous_v"):
        _asai._assert_reference_alignment_matches(_report({"nu": 1, "mu": 2}), _args(True))


def test_missing_flag_in_report_counts_as_old_alignment():
    # Reports written before the flag existed carry no key at all; treating that
    # as "unknown, therefore fine" is what let E7 through.
    _asai._assert_reference_alignment_matches(_report({"nu": 1, "mu": 2}), _args(False))
    with pytest.raises(SystemExit):
        _asai._assert_reference_alignment_matches(_report({}), _args(True))


def test_old_alignment_run_against_a_valigned_report_is_also_refused():
    with pytest.raises(SystemExit, match="contemporaneous_v"):
        _asai._assert_reference_alignment_matches(_report({"contemporaneous_v": True}), _args(False))


@pytest.mark.parametrize("key,value", [("nu", 2), ("mu", 3)])
def test_state_order_mismatch_is_refused(key, value):
    config = {"nu": 1, "mu": 2, "contemporaneous_v": True}
    config[key] = value
    with pytest.raises(SystemExit, match=key):
        _asai._assert_reference_alignment_matches(_report(config), _args(True))
