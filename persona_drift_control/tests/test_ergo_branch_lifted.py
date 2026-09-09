"""Tests for scripts/fit_koopman_ergo_branch_lifted.py.

The load-bearing one is test_policy_separability_detects_a_planted_state_dependent_gain:
the script's headline negative result is "the identified operator wants the
same schedule for every trajectory", and that reading is worthless unless the
checker demonstrably reports MORE than one schedule when the operator really
does have a state-dependent gain. Same discipline as
tests/test_run_config_guard.py: a guard that passes silently hides exactly the
failure it exists to catch.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fit_koopman_ergo_branch_lifted",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fit_koopman_ergo_branch_lifted.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _row(turn: int, msg: str, closeness: float, num_shards: int = 6, refusal: bool = False) -> dict:
    return {"turn": turn, "num_shards": num_shards, "agent_message": msg,
            "closeness": closeness, "refusal_flag": refusal}


def test_lift_marks_the_placeholder_answer():
    psi = mod.lift(_row(3, "I need more info.\nCurrent answer: 0", 0.5), None)
    assert psi[mod.LIFT_NAMES.index("placeholder")] == 1.0
    assert psi[mod.LIFT_NAMES.index("no_number")] == 0.0
    assert psi[0] == 0.5


def test_lift_marks_a_real_answer_and_a_missing_one():
    real = mod.lift(_row(3, "Current answer: 42", 1.0), None)
    assert real[mod.LIFT_NAMES.index("placeholder")] == 0.0
    assert real[mod.LIFT_NAMES.index("no_number")] == 0.0
    missing = mod.lift(_row(3, "I cannot say.", 0.0), None)
    assert missing[mod.LIFT_NAMES.index("no_number")] == 1.0


def test_echo_coordinates_need_a_previous_turn():
    prev = _row(2, "Current answer: 0", 0.5)
    same = mod.lift(_row(3, "Current answer: 0", 0.5), prev)
    assert same[mod.LIFT_NAMES.index("echo_exact")] == 1.0
    assert same[mod.LIFT_NAMES.index("echo_jaccard")] == pytest.approx(1.0)
    first = mod.lift(_row(1, "Current answer: 0", 0.5), None)
    assert first[mod.LIFT_NAMES.index("echo_exact")] == 0.0
    assert first[mod.LIFT_NAMES.index("echo_jaccard")] == 0.0


def test_shard_frac_coordinate_is_turn_over_num_shards():
    psi = mod.lift(_row(3, "Current answer: 5", 0.9, num_shards=6), None)
    assert psi[mod.SHARD_FRAC_IDX] == pytest.approx(0.5)


def _planted_beta(bilinear_gain: float) -> np.ndarray:
    """Operator over the full lift: psi_next = A psi + B u + N psi u + const,
    with the ONLY state coupling of the input being through `placeholder`.
    Design column order is [1, psi(8), u, psi*u(8)] -> 18 rows."""
    d = len(mod.LIFT_NAMES)
    ph = mod.LIFT_NAMES.index("placeholder")
    beta = np.zeros((1 + d + 1 + d, d))
    for j in range(d):
        beta[1 + j, j] = 0.6  # A = 0.6 I, stable, so the value of a reset decays backwards
    # `placeholder` must decay FASTER than the state it modulates, or the two
    # decays cancel and every state prefers the last turn regardless of the
    # planted gain -- which is a degenerate planting, not a working checker.
    beta[1 + ph, ph] = 0.2
    beta[1 + d, 0] = 0.02  # small state-independent gain
    beta[1 + d + 1 + ph, 0] = bilinear_gain
    return beta


def _psi(placeholder: float) -> np.ndarray:
    psi = np.zeros(len(mod.LIFT_NAMES))
    psi[0] = 0.5
    psi[mod.LIFT_NAMES.index("placeholder")] = placeholder
    return psi


def _rows_for_two_states() -> list[dict]:
    rows = []
    for i, ph in enumerate((0.0, 1.0)):
        rows.append({"trajectory_id": f"t{i}", "turn": 2, "num_shards": 6,
                     "psi_prev": _psi(ph), "u": 0.0})
    return rows


def test_policy_separability_detects_a_planted_state_dependent_gain():
    """A gain that is large and positive only while the agent is still emitting
    the placeholder, and that fades faster than the state it acts on, makes an
    EARLY reset worth more in the placeholder state and a LATE one worth more
    otherwise -- two states must produce two schedules."""
    beta = _planted_beta(bilinear_gain=5.0)
    sep = mod.policy_separability(_rows_for_two_states(), beta, "lift_bilinear", [1])
    assert sep[1]["n_distinct_schedules"] == 2, sep[1]["schedule_counts"]


def test_policy_separability_reports_one_schedule_without_a_planted_gain():
    """The same checker on a state-INDEPENDENT operator: one schedule, and with
    a decaying A it is the last turn -- which is what the real fit returns."""
    beta = _planted_beta(bilinear_gain=0.0)
    sep = mod.policy_separability(_rows_for_two_states(), beta, "lift_bilinear", [1])
    assert sep[1]["n_distinct_schedules"] == 1
    assert sep[1]["modal_schedule_from_end"] == [0]


def test_rollout_truth_overrides_shard_frac_instead_of_predicting_it():
    beta = _planted_beta(bilinear_gain=0.0)
    psi0 = _psi(1.0)
    psi0[mod.SHARD_FRAC_IDX] = 2 / 6
    # A = 0.6 I would decay shard_frac towards 0; the override must keep it on
    # its known deterministic path instead.
    row = [{"psi_prev": psi0, "u": 0.0, "shard_frac": 3 / 6}]
    psi_next = (mod.design(row, "lift_bilinear") @ beta)[0]
    assert psi_next[mod.SHARD_FRAC_IDX] == pytest.approx(0.6 * (2 / 6))
    assert mod.rollout_terminal(psi0, [0], 3, 6, beta, "lift_bilinear") == pytest.approx(0.6 * 0.5)


def test_best_null_is_chosen_per_fold_not_per_row():
    """A per-row minimum over the three nulls would be an oracle no null can be.
    Two folds where a different null wins must still yield exactly one null's
    errors per fold."""
    rng = np.random.default_rng(0)
    rows = []
    for item in range(8):
        for turn in (2, 3, 4):
            for u in (0.0, 1.0):
                psi = _psi(float(rng.integers(0, 2)))
                psi[mod.SHARD_FRAC_IDX] = turn / 6
                rows.append({"item_id": f"i{item}", "seed": 0, "trajectory_id": f"t{item}",
                             "turn": turn, "num_shards": 6, "shard_frac": turn / 6,
                             "psi_prev": psi, "u": u,
                             "psi_next": psi, "c_next": 0.4 + 0.1 * u + 0.05 * turn})
    err, groups, per_fold = mod.cross_val_errors(rows, 1e-6)
    assert len(err["best_null"]) == len(groups)
    for f in per_fold:
        assert f["best_null"] in ("constant", "turn_mean", "stateless_ols")
        assert f[f["best_null"]] == min(f["constant"], f["turn_mean"], f["stateless_ols"])
