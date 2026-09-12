"""Tests for scripts/analyze_sequor_targeted_branch.py.

Each quantity gets a planted positive control (an effect the gate must recover
to the digit) and a planted negative control (a defect the gate must refuse).
A gate that passes silently hides the very failure it exists to catch --
the standing rule behind `run_config_guard` and its meta-test.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_sequor_targeted_branch.py"

N_ITEMS = 8
SEEDS = [0, 1, 2]
TURNS = [5, 9, 14]


def _load():
    spec = importlib.util.spec_from_file_location("_targeted_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load()


@pytest.fixture(scope="module")
def gates(module):
    return module._sibling_module("analyze_sequor_s0_0_gates.py")


def _arm(targeted_recovery: float, blanket_recovery: float, held_gap: float = 0.0):
    """A synthetic arm where the recovery rates are what they were planted as.

    One constraint is broken at t-1 on every branch point, so the planted
    recovery rate is exactly the share of branch points whose targeted (or
    blanket) row has that constraint followed at t. The two held constraints
    carry `held_gap` the same way. Nothing here is sampled: the point estimate
    must come back as the planted number, not near it.
    """

    rows, meta = [], {}
    points = [(f"item{i}", seed, turn) for i in range(N_ITEMS) for seed in SEEDS for turn in TURNS]
    for index, (item_id, seed, turn) in enumerate(points):
        share = (index + 0.5) / len(points)
        for branch, recovery, held in ((module_names := ("targeted_remind", "blanket_remind"))[0],
                                       targeted_recovery, 1.0), \
                                      (module_names[1], blanket_recovery, 1.0 - held_gap):
            trajectory_id = f"{item_id}__{branch}__s{seed}__t{turn}"
            followed = [share < recovery, share < held, share < held]
            rows.append({"trajectory_id": trajectory_id, "item_id": item_id, "turn": turn,
                         "branch": branch, "seed": seed, "followed": followed,
                         "y_graded": sum(followed) / 3, "inserted_tokens": 21 if branch == module_names[0] else 41})
            meta[trajectory_id] = {"item_id": item_id, "seed": seed, "turn": turn, "branch": branch,
                                   "violated_indices": [0], "n_violated_at_t_minus_1": 1,
                                   "base_trajectory_id": f"{item_id}__zero_control__s{seed}"}
    return rows, meta


def test_planted_recovery_gap_is_recovered_exactly(module, gates):
    rows, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.50)
    units, drops = module.paired_units(rows, meta, "violated")
    assert drops == {"branch_points_dropped_unparsed": 0, "branch_points_dropped_incomplete": 0}
    assert len(units) == N_ITEMS * len(SEEDS) * len(TURNS)

    block = module.contrast(gates, units, "delta", seed=0)
    assert block["point"] == pytest.approx(0.25, abs=1e-12)
    levels = module.rate_by_branch(units)
    assert levels["targeted"] == pytest.approx(0.75, abs=1e-12)
    assert levels["blanket"] == pytest.approx(0.50, abs=1e-12)


def test_no_planted_gap_gives_exactly_zero(module, gates):
    """The negative control. A pipeline that manufactures a difference out of
    the pairing would show it here, where both branches are the same rows."""

    rows, meta = _arm(targeted_recovery=0.60, blanket_recovery=0.60)
    units, _ = module.paired_units(rows, meta, "violated")
    block = module.contrast(gates, units, "delta", seed=0)
    assert block["point"] == 0.0
    assert block["ci"][0] <= 0.0 <= block["ci"][1]
    assert not block["ci_excludes_zero"]


def test_held_constraints_are_the_complement_not_the_target(module, gates):
    """The secondary quantity must read the constraints the targeted reminder
    did NOT name. Planting a loss there and a gain on the target proves the two
    unit sets are not the same rows under different names."""

    # The planted rates must be multiples of 1/(items*seeds*turns), or the
    # fixture cannot hit them exactly and the test would be checking rounding.
    rows, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.25, held_gap=0.25)
    violated, _ = module.paired_units(rows, meta, "violated")
    held, _ = module.paired_units(rows, meta, "held")
    assert len(held) == 2 * len(violated)
    assert all(unit["constraint_index"] == 0 for unit in violated)
    assert {unit["constraint_index"] for unit in held} == {1, 2}

    assert module.contrast(gates, violated, "delta", seed=0)["point"] == pytest.approx(0.50, abs=1e-12)
    assert module.contrast(gates, held, "delta", seed=0)["point"] == pytest.approx(0.25, abs=1e-12)


def test_a_branch_point_with_an_unparsed_side_is_dropped_whole(module):
    rows, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.50)
    victim = next(row for row in rows if row["branch"] == "blanket_remind")
    victim["followed"] = [True, None, True]
    units, drops = module.paired_units(rows, meta, "violated")
    assert drops["branch_points_dropped_unparsed"] == 1
    assert len(units) == N_ITEMS * len(SEEDS) * len(TURNS) - 1
    key = (victim["item_id"], victim["seed"], victim["turn"])
    assert all((unit["item_id"], unit["seed"], unit["turn"]) != key for unit in units)


def test_a_branch_point_missing_its_other_side_is_dropped(module):
    rows, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.50)
    orphan = next(row for row in rows if row["branch"] == "targeted_remind")
    rows = [row for row in rows
            if not (row["branch"] == "blanket_remind" and row["item_id"] == orphan["item_id"]
                    and row["seed"] == orphan["seed"] and row["turn"] == orphan["turn"])]
    _, drops = module.paired_units(rows, meta, "violated")
    assert drops["branch_points_dropped_incomplete"] == 1


def test_targets_disagreeing_with_the_base_readout_are_refused(module):
    """The arm and the readout on disk must be the pair that produced each
    other. A recorded target the base readout does not produce means every
    contrast below would be computed against the wrong counterfactual."""

    _, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.50)
    verdicts = {(record["base_trajectory_id"], record["turn"] - 1): [False, True, True]
                for record in meta.values()}
    assert module.verify_targets_against_base(meta, verdicts)["n_mismatches"] == 0

    drifted = dict(verdicts)
    key = next(iter(drifted))
    drifted[key] = [True, False, True]
    with pytest.raises(SystemExit, match="not a matched pair"):
        module.verify_targets_against_base(meta, drifted)


def test_an_unparsed_previous_turn_must_not_carry_a_target(module):
    """`violated_indices` is None when t-1 did not parse -- acting on a
    partially parsed turn would let the target depend on which verdict failed."""

    _, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.50)
    verdicts = {(record["base_trajectory_id"], record["turn"] - 1): [False, True, True]
                for record in meta.values()}
    key = next(iter(verdicts))
    verdicts[key] = [None, True, True]
    with pytest.raises(SystemExit, match="not a matched pair"):
        module.verify_targets_against_base(meta, verdicts)


def test_verdict_needs_both_halves_not_just_significance(module):
    """K3's lesson: a CI excluding zero on an effect under the design's own MDE
    is a point estimate detected at less than 80% power. It does not pass."""

    under = {"point": 0.03, "ci": [0.005, 0.055], "ci_excludes_zero": True,
             "mde_at_80pct_this_arm": 0.0525}
    assert module.verdict(under)["verdict"] == "UNDECIDABLE"

    over = {"point": 0.09, "ci": [0.04, 0.14], "ci_excludes_zero": True,
            "mde_at_80pct_this_arm": 0.0525}
    assert module.verdict(over)["verdict"] == "PASS"
    assert module.verdict(over)["closes_koopman_mpc"] is False

    negative = {"point": -0.09, "ci": [-0.14, -0.04], "ci_excludes_zero": True,
                "mde_at_80pct_this_arm": 0.0525}
    assert module.verdict(negative)["verdict"] == "FAIL"
    assert module.verdict(negative)["closes_koopman_mpc"] is True


def test_undecidable_does_not_silently_close_the_line(module):
    """Item 13 signed consequences for PASS and FAIL only. The script must hand
    the third case back rather than default one way and call it pre-registered."""

    crossing = {"point": 0.004, "ci": [-0.03, 0.04], "ci_excludes_zero": False,
                "mde_at_80pct_this_arm": 0.0525}
    ruling = module.verdict(crossing)
    assert ruling["verdict"] == "UNDECIDABLE"
    assert ruling["closes_koopman_mpc"] is None
    assert "NEEDS A USER RULING" in ruling["consequence"]
    assert "NOT evidence that targeting does not work" in ruling["reason"]


def test_bootstrap_resamples_items_not_constraint_observations(module, gates):
    """One item carrying an extreme value must move the interval. If the
    bootstrap resampled the 1104*k constraint observations instead, the
    interval would shrink by the number of turns per dialogue -- turns inside
    one dialogue are not independent draws."""

    rows, meta = _arm(targeted_recovery=0.50, blanket_recovery=0.50)
    units, _ = module.paired_units(rows, meta, "violated")
    for unit in units:
        unit["delta"] = 1.0 if unit["item_id"] == "item0" else 0.0
    block = module.contrast(gates, units, "delta", seed=0)
    assert block["point"] == pytest.approx(1 / N_ITEMS)
    assert block["ci"][0] == pytest.approx(0.0, abs=1e-9)
    assert block["ci"][1] > 2 / N_ITEMS


def test_cost_block_reports_both_branches(module):
    rows, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.50)
    primary = {"point": 0.01, "mde_at_80pct_this_arm": 0.0525}
    cost = module.cost_block(rows, meta, primary)
    assert cost["inserted_tokens_median"]["targeted_remind"] == 21
    assert cost["inserted_tokens_median"]["blanket_remind"] == 41
    assert cost["blanket_over_targeted"] == pytest.approx(41 / 21)


def test_all_k_broken_leaves_no_held_constraints(module, gates):
    """The built-in null in the exploratory breakdown: when every constraint is
    broken, the targeted block IS the blanket block, so there is no unnamed
    constraint to lose and the retention cell must be empty rather than zero."""

    rows, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.25)
    for record in meta.values():
        record["violated_indices"] = [0, 1, 2]
        record["n_violated_at_t_minus_1"] = 3
    violated, _ = module.paired_units(rows, meta, "violated")
    held, _ = module.paired_units(rows, meta, "held")
    assert held == []

    block = module.by_n_violated(gates, violated, held, seed=0)
    assert block["3"]["retention_of_held"]["n_units"] == 0
    assert block["3"]["retention_of_held"]["point"] is None
    assert block["3"]["recovery_of_violated"]["n_units"] == 3 * len(violated) // 3


def test_exploratory_breakdown_is_labelled_not_a_result(module, gates):
    rows, meta = _arm(targeted_recovery=0.75, blanket_recovery=0.25)
    violated, _ = module.paired_units(rows, meta, "violated")
    held, _ = module.paired_units(rows, meta, "held")
    assert "EXPLORATORY" in module.by_n_violated(gates, violated, held, seed=0)["status"]
