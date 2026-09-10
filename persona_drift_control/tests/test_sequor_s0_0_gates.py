"""Tests for scripts/analyze_sequor_s0_0_gates.py, on synthetic readouts.

Positive and negative controls, in the discipline EK-A established: a gate
that can only ever report "no effect" is worthless, so K2 is tested against
planted state dependence (must PASS), against a turn-only ramp (must FAIL),
and against the two degeneracies that make a crossing CI meaningless -- a flat
x-axis and an MDE bigger than the whole mean effect (must be UNDECIDABLE, not
FAIL, because FAIL closes the line).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_sequor_s0_0_gates.py"
_SPEC = importlib.util.spec_from_file_location("analyze_sequor_s0_0_gates", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

Y_LEVELS = (0.0, 1 / 3, 2 / 3, 1.0)


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(mod, "N_BOOTSTRAP", 400)


def _followed(y: float) -> list[bool]:
    n = int(round(y * 3))
    return [True] * n + [False] * (3 - n)


def _snap(value: float) -> float:
    return min(Y_LEVELS, key=lambda level: abs(level - value))


def synth_readout(
    n_items: int = 12, n_turns: int = 20, judge_kind: str = "independent",
    state_slope: float = 0.0, turn_slope: float = 0.0, base_gain: float = 0.0,
    noise: float = 0.0, y_prev_spread: bool = True, decaying: bool = True,
    mid_range: bool = False, seed: int = 0,
) -> dict:
    """A scored arm.

    The base branch gives each item its own starting level and its own decay
    rate, so `y` has cross-trajectory spread at EVERY turn (what K1 asks for)
    while still correlating with the turn index (what makes K2's attenuation
    check meaningful -- a turn ramp leaking through the state is exactly how
    ERGO's marginal slope looked real). `mid_range` keeps `y` away from 0 and 1
    so a planted null gain is not biased by clipping at the boundaries.
    """

    rng = np.random.default_rng(seed)
    starts = [(1 / 3, 2 / 3)[i % 2] if mid_range else Y_LEVELS[i % 4] for i in range(n_items)]
    decays = np.linspace(0.005, 0.030, n_items) if decaying else np.zeros(n_items)
    rows = []
    for item in range(n_items):
        for turn in range(1, n_turns + 1):
            if y_prev_spread:
                y = _snap(float(np.clip(
                    starts[item] - decays[item] * (turn - 1) + 0.08 * rng.standard_normal(), 0.0, 1.0)))
            else:
                y = 2 / 3
            prefix = f"{item:02d}-{turn:02d}"
            rows.append({
                "trajectory_id": f"i{item}__base", "item_id": f"i{item}", "turn": turn,
                "branch": "base", "u_remind": 0, "prefix_sha256": prefix,
                "followed": _followed(y), "y_graded": y, "y_binary": y == 1.0,
                "n_judge_parse_failures": 0, "echo_jaccard_prev": 0.3,
                "inserted_tokens": 0, "hit_token_cap": False,
            })
            if turn == 1:
                continue
            y_prev = rows[-2]["y_graded"]
            gain = base_gain + state_slope * y_prev + turn_slope * turn + noise * rng.standard_normal()
            reminded = float(np.clip(y + gain, 0.0, 1.0))
            rows.append({
                "trajectory_id": f"i{item}__reminded__t{turn}", "item_id": f"i{item}", "turn": turn,
                "branch": "reminded", "u_remind": 1, "prefix_sha256": prefix,
                "followed": _followed(_snap(reminded)), "y_graded": reminded,
                "y_binary": reminded == 1.0, "n_judge_parse_failures": 0,
                "echo_jaccard_prev": 0.5, "inserted_tokens": 40, "hit_token_cap": False,
            })
    return {
        "mode": "canonical", "arm_dir": "outputs/fake", "agent_model": "stub/agent",
        "judge_model": "stub/judge", "judge_kind": judge_kind, "k_constraints": 3,
        "rows": rows,
    }


def test_pairs_are_extracted_with_the_state_before_the_action():
    pairs = mod.pairs_from(synth_readout(n_items=2, n_turns=5))
    assert len(pairs) == 2 * 4
    assert all(p["turn"] >= 2 for p in pairs)
    assert all(p["delta_y"] == p["y_reminded"] - p["y_base"] for p in pairs)


def test_a_prefix_mismatch_aborts():
    report = synth_readout(n_items=2, n_turns=3)
    next(r for r in report["rows"] if r["branch"] == "reminded")["prefix_sha256"] = "tampered"
    with pytest.raises(SystemExit, match="not a causal gain"):
        mod.pairs_from(report)


def test_rows_with_an_undefined_readout_are_dropped_not_zeroed():
    report = synth_readout(n_items=2, n_turns=4)
    victim = next(r for r in report["rows"] if r["branch"] == "reminded" and r["turn"] == 3)
    victim["y_graded"] = None
    pairs = mod.pairs_from(report)
    assert not [p for p in pairs if p["turn"] == 3 and p["item_id"] == victim["item_id"]]


def test_debug_artifacts_are_refused(tmp_path):
    report = synth_readout(n_items=2, n_turns=3)
    report["mode"] = "debug"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="not evidence"):
        mod.load_readout(path, "independent")


def test_a_self_judged_readout_cannot_stand_in_for_the_independent_one(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(synth_readout(n_items=2, n_turns=3, judge_kind="self")))
    with pytest.raises(SystemExit, match="expected 'independent'"):
        mod.load_readout(path, "independent")


def test_k1_passes_on_a_readout_with_range():
    k1 = mod.gate_k1(synth_readout())
    assert k1["pass"] and not k1["turns_failing"] and k1["n_patterns_seen"] >= 4


def test_k1_fails_when_an_in_scope_turn_is_saturated():
    """The `defense` line's death: a turn at mean 1.000, sd 0.000, one distinct
    value. It must be caught here, before anything is fitted."""

    report = synth_readout()
    for row in report["rows"]:
        if row["turn"] == 3:
            row["y_graded"] = 1.0
            row["followed"] = [True, True, True]
    k1 = mod.gate_k1(report)
    assert not k1["pass"] and 3 in k1["turns_failing"]
    assert k1["per_turn"][3]["sd"] == 0.0 and k1["per_turn"][3]["ceiling_share"] == 1.0


def test_a_saturated_turn_one_does_not_decide_k1_but_is_still_reported():
    """Scope t2.. (user ruling 2026-09-09): turn 1 states the constraints, so it
    admits no u=1 action and contributes no pair to K2/K3. It must not close the
    line on its own -- and it must still be visible in the artifact, or the
    scope would be a way of hiding a saturated readout."""

    report = synth_readout()
    for row in report["rows"]:
        if row["turn"] == 1:
            row["y_graded"] = 1.0
            row["followed"] = [True, True, True]
    k1 = mod.gate_k1(report)
    assert k1["pass"] and 1 not in k1["turns_failing"]
    assert k1["per_turn_out_of_scope"][1]["sd"] == 0.0
    assert k1["per_turn_out_of_scope"][1]["ceiling_share"] == 1.0
    assert k1["turn_scope"].startswith("t2")


def test_k2_recovers_planted_state_dependence():
    pairs = mod.pairs_from(synth_readout(state_slope=-0.30, base_gain=0.35, noise=0.02,
                                        mid_range=True, seed=1))
    k2 = mod.gate_k2(pairs, seed=0)
    assert k2["verdict"] == "PASS"
    assert k2["ci"][0] < 0 and k2["ci"][1] < 0
    assert k2["slope_on_y_prev"] == pytest.approx(-0.30, abs=0.10)


def test_k2_fails_when_the_gain_follows_only_the_turn_index():
    """ERGO's death, planted: attenuation must expose that the marginal slope
    was the turn ramp leaking through the state's correlation with it."""

    pairs = mod.pairs_from(synth_readout(turn_slope=-0.012, base_gain=0.40, noise=0.01,
                                        mid_range=True, seed=2))
    k2 = mod.gate_k2(pairs, seed=0)
    assert k2["verdict"] == "FAIL"
    assert abs(k2["slope_on_y_prev"]) < 0.10
    assert k2["attenuation"] is None or k2["attenuation"] > 0.0


def test_k2_is_undecidable_when_the_x_axis_is_flat():
    """EK-A's trap: `c_prev` had IQR 0 with 75% of rows at one value, so the
    regression had no x-axis. A crossing CI there says nothing."""

    pairs = mod.pairs_from(synth_readout(base_gain=0.30, noise=0.01, y_prev_spread=False, seed=3))
    k2 = mod.gate_k2(pairs, seed=0)
    assert k2["verdict"] == "UNDECIDABLE"
    assert k2["degenerate_axis"] and k2["y_prev_iqr"] == 0.0


def test_k2_is_undecidable_when_the_design_cannot_resolve_the_effect():
    """The power clause on its own branch: the CI must ALSO cross zero. Without
    that assertion this test would still pass under the buggy ordering, so it
    would not be guarding what it claims to guard.

    The fixture was retuned when that assertion went in (2026-09-09). The
    original one -- same noise, no `mid_range` -- had `y` pinned against 0 and
    1, and the clipping manufactured a slope of -0.44 whose CI EXCLUDED zero.
    It reported UNDECIDABLE only because the buggy clause reached the verdict
    first, so the test was green for the wrong reason. `mid_range` keeps the
    readout off the boundaries, and the unpowered case is then a real one."""

    pairs = mod.pairs_from(synth_readout(state_slope=-0.02, base_gain=0.02, noise=0.60,
                                        mid_range=True, seed=4))
    k2 = mod.gate_k2(pairs, seed=0)
    assert k2["verdict"] == "UNDECIDABLE"
    assert k2["underpowered"] and k2["mde_at_80pct"] > abs(k2["mean_gain_for_scale"])
    assert k2["ci"][0] * k2["ci"][1] <= 0 and not k2["excludes_zero"]


def test_k2_passes_when_the_ci_excludes_zero_even_though_the_mde_exceeds_the_mean_gain():
    """Regression for the clause bug (screening section 10 item 7): the power
    clause used to sit AHEAD of the pass branch, so this exact combination --
    a slope whose CI excludes zero, on a design whose slope MDE is larger than
    the mean gain -- was reported UNDECIDABLE. A CI excluding zero demonstrates
    power after the fact; the clause guards the FAIL branch only.

    `underpowered` is still true here, which is what makes this the case the
    old ordering could not get right."""

    pairs = mod.pairs_from(synth_readout(state_slope=-0.30, base_gain=0.15, noise=0.05,
                                        mid_range=True, seed=1))
    k2 = mod.gate_k2(pairs, seed=0)
    assert k2["underpowered"]
    assert k2["mde_at_80pct"] > abs(k2["mean_gain_for_scale"])
    assert k2["excludes_zero"] and k2["ci"][0] < 0 and k2["ci"][1] < 0
    assert k2["attenuation"] < 0.50
    assert k2["verdict"] == "PASS"


def pairs_with_heterogeneous_slope(slope: float, item_sd: float, noise: float = 0.05,
                                   base_gain: float = 0.30, n_items: int = 12,
                                   n_turns: int = 19, seed: int = 0) -> list[dict]:
    """Pairs built directly, so the slope's own standard error can be dialled
    by the spread of per-item slopes -- the knob `synth_readout` does not
    expose. Used to land a slope that is significant at ~1.96 sigma yet below
    the 80%-power (2.8 sigma) MDE."""

    rng = np.random.default_rng(seed)
    out = []
    for item in range(n_items):
        item_slope = slope + item_sd * rng.standard_normal()
        for turn in range(2, n_turns + 2):
            y_prev = Y_LEVELS[rng.integers(4)]
            out.append({"item_id": f"i{item}", "turn": turn, "y_prev": y_prev,
                        "delta_y": base_gain + item_slope * y_prev + noise * rng.standard_normal()})
    return out


def test_k2_pass_below_its_own_mde_carries_the_inflation_caveat():
    """This is the S0-0 fidelity-harness situation: |slope| 0.86x the 80%-power
    MDE. The gate passes -- it is a go/no-go filter -- but the point estimate
    was detected at under 80% power and is likely inflated, so the verdict must
    carry that in writing, next to the number, or it will be quoted as a
    measurement."""

    k2 = mod.gate_k2(pairs_with_heterogeneous_slope(-0.20, item_sd=0.18, seed=2), seed=0)
    assert k2["verdict"] == "PASS"
    assert k2["excludes_zero"]
    assert 0.5 < k2["slope_over_mde"] < 1.0
    assert "likely inflated" in k2["caveat"]
    assert "never a reported number" in k2["caveat"]


def test_k2_fails_not_undecidable_when_a_resolved_slope_is_mostly_the_turn_ramp():
    """The other side of the reordering: moving the power clause off the pass
    branch must not turn a real FAIL into an escape. A slope whose CI excludes
    zero but which attenuates >= 50% against the marginal fit is ERGO's death,
    and it stays a FAIL."""

    pairs = mod.pairs_from(synth_readout(state_slope=-0.06, turn_slope=-0.02, base_gain=0.45,
                                        noise=0.01, mid_range=True, seed=2))
    k2 = mod.gate_k2(pairs, seed=0)
    assert k2["excludes_zero"] and k2["attenuation"] >= 0.50
    assert k2["verdict"] == "FAIL"
    assert "turn proxy" in k2["reason"]


def test_k3_passes_on_a_real_effect_and_reports_it_against_the_mde():
    pairs = mod.pairs_from(synth_readout(base_gain=0.30, noise=0.05, seed=5))
    k3 = mod.gate_k3(pairs, seed=0)
    assert k3["pass"] and k3["mean_delta_y"] > 0
    assert k3["effect_over_mde"] > 1.0
    assert k3["n_items"] == 12


def test_k3_fails_on_a_null_effect():
    pairs = mod.pairs_from(synth_readout(base_gain=0.0, noise=0.10, mid_range=True,
                                        decaying=False, seed=6))
    k3 = mod.gate_k3(pairs, seed=0)
    assert not k3["pass"]


def test_k3_reports_the_share_of_pairs_that_are_exactly_zero():
    """EK-A's other trap: 87.5% of its pairs had delta exactly 0, so a
    significant mean came from a small minority of turns."""

    pairs = mod.pairs_from(synth_readout(base_gain=0.0, noise=0.0, seed=7))
    assert mod.gate_k3(pairs, seed=0)["share_exactly_zero"] == 1.0


def test_self_vs_independent_is_split_by_action():
    independent = synth_readout(base_gain=0.2, noise=0.02, seed=8)
    self_report = json.loads(json.dumps(independent))
    self_report["judge_kind"] = "self"
    for row in self_report["rows"]:  # a self-judge that is generous on reminded rows only
        if row["u_remind"]:
            row["y_graded"] = min(1.0, row["y_graded"] + 0.2)
    split = mod.judge_agreement_by_action(independent, self_report)
    assert split["u=0 (base)"]["mean_signed_gap_self_minus_independent"] == pytest.approx(0.0)
    assert split["u=1 (reminded)"]["mean_signed_gap_self_minus_independent"] > 0.1


def _arm_report(shares: dict[str, float], criterion: float = 0.05) -> dict:
    return {
        "cap_criterion": criterion, "token_cap_share": sum(shares.values()) / len(shares),
        "max_new_tokens": 2048,
        "token_cap_by_item": {
            item: {"n_rows": 39, "n_hit_token_cap": round(share * 39), "token_cap_share": share,
                   "output_tokens_median": 600, "constraints": ["a", "b", "c"]}
            for item, share in shares.items()
        },
    }


def test_cap_guard_passes_when_no_item_binds():
    record = mod.refuse_if_the_cap_bound(_arm_report({"i0": 0.0, "i1": 0.02}), [])
    assert record["excluded_items"] == [] and record["max_new_tokens"] == 2048


def test_cap_guard_refuses_a_single_offending_item_even_when_the_average_is_fine():
    """Job 15756689's shape: a global average of 18.6% with two items at 92%
    and 100%. A guard on the average would have let those two through."""

    report = _arm_report({f"i{i}": 0.0 for i in range(10)} | {"long_a": 0.92, "long_b": 1.0})
    with pytest.raises(SystemExit, match="long_a 92.0%|long_b 100.0%"):
        mod.refuse_if_the_cap_bound(report, [])


def test_cap_guard_is_passable_only_by_naming_the_excluded_items():
    report = _arm_report({"i0": 0.0, "long_a": 0.92})
    record = mod.refuse_if_the_cap_bound(report, ["long_a"])
    assert record["excluded_items"] == ["long_a"]
    with pytest.raises(SystemExit):
        mod.refuse_if_the_cap_bound(report, ["some_other_item"])


def test_cap_guard_refuses_an_arm_report_that_predates_the_criterion():
    """An older artifact must not read as compliant just because the field it
    would have failed on does not exist yet."""

    with pytest.raises(SystemExit, match="predates"):
        mod.refuse_if_the_cap_bound({"token_cap_share": 0.186}, [])


def test_pairs_are_keyed_by_seed():
    """Under sampled decoding an arm runs each item several times. A pair built
    across seeds would compare two trajectories, not one action."""

    a = synth_readout(n_items=2, n_turns=4, base_gain=0.3, seed=1)
    b = synth_readout(n_items=2, n_turns=4, base_gain=0.3, seed=2)
    for r in a["rows"]:
        r["seed"] = 0
    for r in b["rows"]:
        r["seed"] = 1
        r["prefix_sha256"] = "s1-" + r["prefix_sha256"]
    merged = dict(a)
    merged["rows"] = a["rows"] + b["rows"]
    pairs = mod.pairs_from(merged)
    assert len(pairs) == 2 * 2 * 3          # two seeds x two items x (T-1) pairs
    assert {p["seed"] for p in pairs} == {0, 1}


def test_a_pair_split_across_seeds_is_not_formed():
    report = synth_readout(n_items=1, n_turns=3, base_gain=0.3)
    for r in report["rows"]:
        r["seed"] = 0 if r["branch"] == "base" else 1
    assert mod.pairs_from(report) == []
