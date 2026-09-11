"""Tests for scripts/fit_koopman_sequor_model.py.

Each gate gets a planted positive control (a system it must recognise) and a
planted negative control (a system it must refuse to pass). The negative
controls matter more here: G-S2-1 is the gate ERGO's operator failed, and a
version of it that passes on state-free data would have let this line inherit
ERGO's death without noticing.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fit_koopman_sequor_model.py"

N_ITEMS = 20
N_TURNS = 20
SEEDS = [0, 1, 2]


def _load():
    spec = importlib.util.spec_from_file_location("_sequor_koopman", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _followed(y: float) -> list[bool]:
    """The constraint vector a readout row carries, quantised from `y`.

    The real judge returns `followed: [b, b, b]` and `y_graded` is its mean, so
    a fixture whose `y` moves while its vector does not would let the binary
    model pass on a state that never varies."""

    kept = int(round(float(np.clip(y, 0.0, 1.0)) * 3))
    return [i < kept for i in range(3)]


def _synthetic(a: float, b: float, *, noise: float = 0.02, item_spread: float = 0.0,
               lagged_actuator: bool = False, seed: int = 0) -> list[dict]:
    """Trajectories from a known `y_(t+1) = a y_t + b u_(t+1) + c_item`.

    `u` is drawn per (item, turn, seed) and mirrored between the two arms, the
    way the antithetic design does it. `lagged_actuator=True` instead makes
    the action land one turn late, which is the off-by-one the mandatory
    `--contemporaneous-v` exists to rule out.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for item in range(N_ITEMS):
        offset = item_spread * ((item % 5) - 2) / 2
        for s in SEEDS:
            schedule = {t: int(rng.integers(0, 2)) for t in range(2, N_TURNS + 1)}
            for arm in ("bernoulli", "antithetic"):
                y = 0.5 + offset
                for turn in range(1, N_TURNS + 1):
                    if turn == 1:
                        u = 0
                    else:
                        u = schedule[turn] if arm == "bernoulli" else 1 - schedule[turn]
                    if turn > 1:
                        drive = prev_u if lagged_actuator else u
                        y = a * y + b * drive + (1 - a) * (0.5 + offset) + rng.normal(0, noise)
                    prev_u = u
                    rows.append({
                        "trajectory_id": f"tuple_{item}__{arm}__s{s}", "item_id": f"tuple_{item}",
                        "turn": turn, "branch": arm, "u_remind": u, "seed": s,
                        "y_graded": float(np.clip(y, 0.0, 1.0)),
                        "followed": _followed(float(np.clip(y, 0.0, 1.0))),
                        "echo_jaccard_prev": None,
                        "inserted_tokens": 0, "hit_token_cap": False,
                    })
    return rows


def _s1a_rows() -> list[dict]:
    """Full-dose arms, which must never enter identification: `u` is constant
    within each of them, so they carry no information about `B`."""

    rows = []
    for arm, u in (("zero_control", 0), ("constant_remind", 1)):
        for item in range(N_ITEMS):
            for s in SEEDS:
                for turn in range(1, N_TURNS + 1):
                    rows.append({
                        "trajectory_id": f"tuple_{item}__{arm}__s{s}", "item_id": f"tuple_{item}",
                        "turn": turn, "branch": arm, "u_remind": u if turn > 1 else 0,
                        "seed": s, "y_graded": 0.5, "followed": _followed(0.5),
                        "echo_jaccard_prev": None, "inserted_tokens": 0, "hit_token_cap": False,
                    })
    return rows


def _readout(rows: list[dict], tmp_path: pathlib.Path, *, judge_kind: str = "independent") -> pathlib.Path:
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir(exist_ok=True)
    arm_report = {
        "mode": "canonical", "arms": ["zero_control", "constant_remind", "bernoulli", "antithetic"],
        "seeds": SEEDS, "n_turns": N_TURNS, "n_items": N_ITEMS,
        "cap_criterion": 0.05, "token_cap_share": 0.0, "max_new_tokens": 2048,
        "token_cap_by_item": {f"tuple_{i}": {"token_cap_share": 0.0} for i in range(N_ITEMS)},
        "gold_coverage": {"n_constraints_judged": 60, "n_constraints_in_calibration_set": 21,
                          "share_outside_calibration_set": 0.65},
        "cell_coverage": {"cells_are_matched": True},
        "schedule_checks": {},
    }
    (arm_dir / "arm_report.json").write_text(json.dumps(arm_report))
    path = tmp_path / f"readout_{judge_kind}.json"
    path.write_text(json.dumps({
        "mode": "canonical", "arm_dir": str(arm_dir), "agent_model": "agent", "judge_model": "judge",
        "judge_kind": judge_kind, "n_rows": len(rows), "rows": rows,
    }))
    return path


def _fit(module, rows, *, contemporaneous_v: bool = True, ridge: float = 1e-6):
    from persona_drift.modeling.dataset import ReducedStateConfig, build_identification_dataset

    config = ReducedStateConfig(nu=1, mu=1, contemporaneous_v=contemporaneous_v)
    prepared = module.rows_for_identification({"rows": rows}, [])
    dataset = build_identification_dataset(prepared, config, y_col="y_graded", u_col="u_remind")
    design = module.design_matrix(prepared, config)
    return config, prepared, dataset, design


def test_planted_operator_is_recovered_and_both_gates_pass():
    module = _load()
    defense = module._sibling_module("fit_koopman_defense_model.py")
    rows = _synthetic(a=0.6, b=0.10)
    config, prepared, dataset, design = _fit(module, rows)

    b = module.bootstrap_b(design, 1e-6, seed=0)
    assert b["B"] == pytest.approx(0.10, abs=0.02)
    gate2 = module.gate_s2_2(b)
    assert gate2["verdict"] == "PASS" and gate2["ci_excludes_zero"]

    folds = module.fold_evaluation(defense, prepared, config, 1e-6, n_folds=20, seed=0)
    gate1 = module.gate_s2_1(folds, folds_to_pass=14, n_folds=20)
    assert gate1["verdict"] == "PASS", folds["arx_n_folds_beating_best_null"]


def test_state_free_data_fails_the_state_gate():
    """Negative control: `y_(t+1)` does not depend on `y_t` at all. An arx
    that still "beat" the nulls here would be reading noise, and the gate
    would be unable to tell this line from ERGO's operator."""

    module = _load()
    defense = module._sibling_module("fit_koopman_defense_model.py")
    rows = _synthetic(a=0.0, b=0.10, noise=0.05)
    config, prepared, _, _ = _fit(module, rows)
    folds = module.fold_evaluation(defense, prepared, config, 1e-6, n_folds=20, seed=0)
    gate = module.gate_s2_1(folds, folds_to_pass=14, n_folds=20)
    assert gate["verdict"] == "FAIL", folds["arx_n_folds_beating_best_null"]


def test_a_dead_actuator_fails_the_controllability_gate():
    """Negative control for G-S2-2: `b = 0`, so the CI must cover zero."""

    module = _load()
    rows = _synthetic(a=0.6, b=0.0)
    _, _, _, design = _fit(module, rows)
    gate = module.gate_s2_2(module.bootstrap_b(design, 1e-6, seed=0))
    assert gate["verdict"] == "FAIL"
    assert gate["ci"][0] < 0 < gate["ci"][1]


def test_contemporaneous_alignment_is_what_recovers_b():
    """The off-by-one that cost both earlier lines a retracted verdict: when
    the action lands inside the turn, fitting the lagged slot estimates
    carryover, not the actuator."""

    module = _load()
    rows = _synthetic(a=0.6, b=0.10)
    _, _, _, contemporaneous = _fit(module, rows, contemporaneous_v=True)
    _, _, _, lagged = _fit(module, rows, contemporaneous_v=False)
    b_now = module.bootstrap_b(contemporaneous, 1e-6, seed=0)["B"]
    b_lag = module.bootstrap_b(lagged, 1e-6, seed=0)["B"]
    assert b_now == pytest.approx(0.10, abs=0.02)
    assert abs(b_lag) < 0.5 * abs(b_now)


def test_the_explicit_design_must_agree_with_the_surrogate():
    """The bootstrap runs on an explicit least-squares design; if it ever
    stopped reproducing `KoopmanSurrogate`'s own `B`, the CI would describe a
    model nobody fit. The guard must fire on a tampered design."""

    module = _load()
    rows = _synthetic(a=0.6, b=0.10)
    _, _, dataset, design = _fit(module, rows)
    assert module._assert_matches_surrogate(design, dataset, 1e-6) == pytest.approx(0.10, abs=0.02)

    # A constant shift would only move the intercept; scaling the target moves
    # the coefficient the bootstrap reads.
    tampered = dict(design, y=design["y"] * 1.5)
    with pytest.raises(SystemExit, match="nobody fit"):
        module._assert_matches_surrogate(tampered, dataset, 1e-6)


def test_only_the_excitation_arms_enter_identification():
    module = _load()
    rows = _synthetic(a=0.6, b=0.10) + _s1a_rows()
    prepared = module.rows_for_identification({"rows": rows}, [])
    assert {r["trajectory_id"].split("__")[1] for r in prepared} == set(module.S1B_ARMS)


def test_excluded_items_leave_the_fit():
    module = _load()
    rows = _synthetic(a=0.6, b=0.10)
    prepared = module.rows_for_identification({"rows": rows}, ["tuple_3"])
    assert "tuple_3" not in {r["item_id"] for r in prepared}


def test_folds_are_item_disjoint():
    module = _load()
    defense = module._sibling_module("fit_koopman_defense_model.py")
    rows = _synthetic(a=0.6, b=0.10)
    config, prepared, _, _ = _fit(module, rows)
    summary = module.fold_evaluation(defense, prepared, config, 1e-6, n_folds=20, seed=0)
    held = [set(f["held_out_items"]) for f in summary["folds"]]
    assert sum(len(h) for h in held) == N_ITEMS
    assert set().union(*held) == {f"tuple_{i}" for i in range(N_ITEMS)}
    for i, a in enumerate(held):
        for b in held[i + 1:]:
            assert not (a & b)


def test_an_unparsed_verdict_drops_its_transitions_instead_of_splicing():
    """A None readout becomes NaN and the builder drops every pair touching
    it. Dropping the ROW instead would glue turn t-1 onto turn t+1 as if they
    were adjacent -- a fabricated transition."""

    module = _load()
    rows = _synthetic(a=0.6, b=0.10)
    _, _, _, before = _fit(module, rows)
    for row in rows:
        if row["item_id"] == "tuple_0" and row["turn"] == 10:
            row["y_graded"] = None
    _, _, _, after = _fit(module, rows)
    lost = before["X"].shape[0] - after["X"].shape[0]
    assert lost == 2 * len(SEEDS) * 2, lost  # two transitions per affected trajectory


def test_state_provenance_separates_dynamics_from_item_difficulty():
    """Planted control: no dynamics at all, only a per-item level. The pooled
    coefficient must look large and the item-demeaned one must collapse --
    otherwise the diagnostic could not tell the two apart on real rows."""

    module = _load()
    rng = np.random.default_rng(0)
    rows = []
    for item in range(N_ITEMS):
        level = 0.2 + 0.03 * item
        for s in SEEDS:
            for turn in range(1, N_TURNS + 1):
                # white noise around a per-item level: no dynamics to find, and
                # unlike a deterministic cycle it has no autocorrelation of its own
                rows.append({
                    "trajectory_id": f"tuple_{item}__bernoulli__s{s}", "item_id": f"tuple_{item}",
                    "turn": turn, "branch": "bernoulli", "u_remind": float(rng.integers(0, 2)),
                    "seed": s, "y_graded": level + float(rng.normal(0, 0.01)),
                })
    prepared = module.rows_for_identification({"rows": rows}, [])
    prov = module.state_provenance(prepared)
    assert prov["is_a_gate"] is False
    assert prov["pooled"]["y_prev_coefficient"] > 0.8
    assert abs(prov["item_demeaned"]["y_prev_coefficient"]) < 0.3

    dynamic = module.rows_for_identification({"rows": _synthetic(a=0.6, b=0.10, item_spread=0.0)}, [])
    assert module.state_provenance(dynamic)["item_demeaned"]["y_prev_coefficient"] > 0.4


def test_end_to_end_refuses_to_overwrite_and_writes_every_gate(tmp_path, monkeypatch):
    module = _load()
    # the sizing gate reads the S1a arms, so the end-to-end fixture carries all four
    path = _readout(_synthetic(a=0.6, b=0.10) + _sizing_rows()[0], tmp_path)
    out = tmp_path / "s2_report.json"
    monkeypatch.setattr("sys.argv", [
        "fit_koopman_sequor_model.py", "--independent-readout", str(path),
        "--contemporaneous-v", "--out-path", str(out), "--n-folds", "10", "--folds-to-pass", "7",
    ])
    module.main()
    report = json.loads(out.read_text())
    assert report["g_s2_1"]["verdict"] == "PASS"
    assert report["g_s2_2"]["verdict"] == "PASS"
    assert report["g_s2_4"]["verdict"] == "PASS"
    assert report["g_s2_4"]["target_fraction"] == 0.5
    assert report["g_s2_3"]["full_rank"] is True
    assert "does not close" in report["caveat"] or "never used to close" in report["caveat"]

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        module.main()


def test_the_self_readout_cannot_feed_the_fit(tmp_path):
    module = _load()
    gates = module._sibling_module("analyze_sequor_s0_0_gates.py")
    path = _readout(_synthetic(a=0.6, b=0.10), tmp_path, judge_kind="self")
    with pytest.raises(SystemExit, match="judge_kind"):
        gates.load_readout(path, "independent")


def _sizing_rows():
    """Two S1a arms with a known full-dose contrast and a known paired
    variance: `constant_remind` sits about `gap` above `zero_control` in every
    late turn, item LEVELS spread widely while the per-item EFFECT barely
    moves. Pairing removes the level, so the two calibers differ by an order
    of magnitude -- which is the thing under test."""

    rng = np.random.default_rng(7)
    gap = 0.20
    rows = []
    for item in range(N_ITEMS):
        # wide spread in LEVEL, narrow spread in the per-item EFFECT: exactly the
        # shape that makes the paired and unpaired calibers disagree
        level = 0.1 + 0.04 * item
        item_gap = gap + 0.02 * ((item % 5) - 2)
        for s in SEEDS:
            for arm in ("zero_control", "constant_remind"):
                for turn in range(1, N_TURNS + 1):
                    y = level + (item_gap if arm == "constant_remind" else 0.0) + rng.normal(0, 0.01)
                    rows.append({
                        "trajectory_id": f"tuple_{item}__{arm}__s{s}", "item_id": f"tuple_{item}",
                        "turn": turn, "branch": arm, "u_remind": int(arm == "constant_remind" and turn > 1),
                        "seed": s, "y_graded": float(y), "followed": _followed(float(y)),
                    })
    return rows, gap


def test_sizing_uses_the_paired_caliber_not_the_unpaired_one():
    """The caliber correction signed 2026-09-10. Between-item spread dominates
    this readout, so the unpaired sd of late-window means is much larger than
    the sd of the PAIRED per-(item, seed) difference that plan section 6's
    primary is built on. Sizing on the unpaired number prices a design nobody
    runs -- here it would demand an order of magnitude more items."""

    module = _load()
    pilot = module._sibling_module("analyze_sequor_s1_pilot.py")
    rows, gap = _sizing_rows()
    sizing = module.s3_sizing_variance(pilot, rows, late_from=15, n_turns=N_TURNS)

    unpaired = np.std([np.mean([r["y_graded"] for r in rows
                                if r["item_id"] == f"tuple_{i}" and r["seed"] == s
                                and r["branch"] == arm and r["turn"] >= 15])
                       for i in range(N_ITEMS) for s in SEEDS
                       for arm in ("zero_control", "constant_remind")], ddof=1)
    paired = np.hypot(sizing["sd_between_items"], sizing["sd_within_item_across_seeds"])
    assert paired < 0.2 * unpaired, (paired, unpaired)


def test_the_signed_target_is_half_the_full_dose_contrast():
    module = _load()
    pilot = module._sibling_module("analyze_sequor_s1_pilot.py")
    rows, gap = _sizing_rows()
    sizing = module.s3_sizing_variance(pilot, rows, late_from=15, n_turns=N_TURNS)
    gate = module.gate_s2_4(gap, sizing, seeds=3, fraction=0.5, unpaired_late_sd=0.99, b=0.02)

    assert gate["target_effect"] == pytest.approx(gap / 2)
    assert gate["verdict"] == "PASS"
    assert gate["trajectories_per_arm"] == gate["n_items_required"] * 3
    # the superseded number stays in the artifact, labelled
    assert gate["unpaired_late_window_sd_superseded"] == 0.99
    # and the cost of the ruling is recorded with it
    assert gate["sizing_table"]["0.25x_full_dose"]["n_items"] > gate["n_items_required"]
    assert any("UNDECIDABLE" in rule for rule in gate["s3_reporting_rules_signed_with_this"])


def test_an_unaffordable_target_stops_instead_of_passing():
    """The ceiling is a stop rule, not a suggestion: a target the design cannot
    afford must hand back, never quietly size past 150 trajectories per arm."""

    module = _load()
    pilot = module._sibling_module("analyze_sequor_s1_pilot.py")
    rows, gap = _sizing_rows()
    sizing = module.s3_sizing_variance(pilot, rows, late_from=15, n_turns=N_TURNS)
    sizing = dict(sizing, sd_between_items=sizing["sd_between_items"] * 10)
    gate = module.gate_s2_4(gap, sizing, seeds=3, fraction=0.5, unpaired_late_sd=0.99, b=0.02)
    assert gate["verdict"] == "STOP_AND_REPORT"
    assert gate["trajectories_per_arm"] > module.S3_TRAJECTORIES_PER_ARM_CEILING


def test_seeds_buy_less_than_items_in_the_sizing():
    """Between-item variance does not shrink with seeds. If the sizing ever
    stopped reflecting that, someone would buy seeds expecting power only
    items can deliver -- the error S1's own sizing was written to prevent."""

    module = _load()
    pilot = module._sibling_module("analyze_sequor_s1_pilot.py")
    rows, gap = _sizing_rows()
    sizing = module.s3_sizing_variance(pilot, rows, late_from=15, n_turns=N_TURNS)
    three = module.gate_s2_4(gap, sizing, seeds=3, fraction=0.25, unpaired_late_sd=0.9, b=0.02)
    nine = module.gate_s2_4(gap, sizing, seeds=9, fraction=0.25, unpaired_late_sd=0.9, b=0.02)
    assert nine["n_items_required"] >= 0.8 * three["n_items_required"]


def _interaction_rows(d: float, *, a: float = 0.6, b: float = 0.10, noise: float = 0.02,
                      seed: int = 3) -> list[dict]:
    """Trajectories from `y_(t+1) = c + a y_t + u (b + d y_t)`."""

    rng = np.random.default_rng(seed)
    rows = []
    for item in range(N_ITEMS):
        for s in SEEDS:
            schedule = {t: int(rng.integers(0, 2)) for t in range(2, N_TURNS + 1)}
            for arm in ("bernoulli", "antithetic"):
                y = 0.5
                for turn in range(1, N_TURNS + 1):
                    u = 0 if turn == 1 else (schedule[turn] if arm == "bernoulli" else 1 - schedule[turn])
                    if turn > 1:
                        y = a * y + u * (b + d * y) + (1 - a) * 0.5 + rng.normal(0, noise)
                    rows.append({
                        "trajectory_id": f"tuple_{item}__{arm}__s{s}", "item_id": f"tuple_{item}",
                        "turn": turn, "branch": arm, "u_remind": u, "seed": s,
                        "y_graded": float(np.clip(y, 0.0, 1.0)),
                    })
    return rows


def test_the_bilinear_term_is_recovered_when_it_is_there():
    module = _load()
    rows = module.rows_for_identification({"rows": _interaction_rows(d=-0.30)}, [])
    gate = module.fit_bilinear(module.one_step_transitions(rows), seed=0)
    assert gate["pooled"]["interaction_d"] == pytest.approx(-0.30, abs=0.06)
    assert gate["verdict"] == "PASS"
    marginal = gate["pooled"]["marginal_effect_of_a_reminder"]
    assert marginal["y=0.00"] > marginal["y=1.00"]


def test_no_interaction_is_not_reported_as_one():
    """Negative control. ERGO's bilinear term was null and enumeration then
    showed the closed loop was degenerate; a fit that invented an interaction
    would send this line into the same 6 GPU-hours ERGO avoided."""

    module = _load()
    rows = module.rows_for_identification({"rows": _interaction_rows(d=0.0)}, [])
    gate = module.fit_bilinear(module.one_step_transitions(rows), seed=0)
    assert gate["pooled"]["interaction_d"] == pytest.approx(0.0, abs=0.05)
    assert gate["verdict"] in ("NULL", "UNDECIDABLE")
    assert not gate["pooled"]["ci_excludes_zero"]


def test_an_interior_operator_wants_one_schedule_for_everyone():
    """The structural fact S3 hinges on, as a test rather than an argument: on
    a scalar operator whose trajectories stay inside [0, 1], the budget-k
    optimum is the same k turns at EVERY starting state, with or without a
    bilinear term. This is the negative control for the checker and, on the
    real fit, the finding itself."""

    module = _load()
    starts = {f"y{v:.2f}": float(v) for v in np.linspace(0.1, 0.9, 17)}
    linear = {"a": 0.63, "b": 0.018, "c": 0.31, "d": 0.0}
    bilinear = {**linear, "b": 0.072, "d": -0.061}
    out = module.schedule_separability({"linear": linear, "bilinear": bilinear}, starts,
                                       (1, 2, 3), start_turn=2, n_turns=20, late_from=15)
    assert out["negative_control_holds"]
    assert not out["closed_loop_has_something_to_do"]
    assert all(v == 1 for v in out["bilinear_distinct_schedules"].values())


def test_a_saturating_operator_does_want_different_schedules():
    """Positive control: the checker must be ABLE to report state-dependent
    schedules, or its "1 distinct" finding would be unfalsifiable. Saturation
    is the honest way to produce one -- reminding a trajectory already at the
    ceiling is wasted, so the best turn depends on where the state is. It is
    also not hypothetical here: 60-67% of real late turns sit at y = 1, a
    censoring the linear fit smooths away."""

    module = _load()
    starts = {f"y{v:.2f}": float(v) for v in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)}
    saturating = {"a": 0.9, "b": 0.30, "c": 0.02, "d": -0.50}
    out = module.schedule_separability({"bilinear": saturating}, starts, (2, 3),
                                       start_turn=2, n_turns=20, late_from=15)
    assert out["closed_loop_has_something_to_do"]
    assert max(out["bilinear_distinct_schedules"].values()) > 1


def test_the_simulation_ranks_feedback_above_random_and_uses_common_noise():
    module = _load()
    starts = {f"y{v:.2f}": float(v) for v in np.linspace(0.2, 0.8, 8)}
    op = {"a": 0.63, "b": 0.072, "c": 0.31, "d": -0.061}
    residuals = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    sim = module.simulate_s3_gap(op, starts, residuals, budget=3, start_turn=2, n_turns=20,
                                 late_from=15, n_sims=20, seed=0)
    assert sim["is_a_gate"] is False
    # feedback must beat a random placement, or the simulator is not measuring control at all
    assert sim["mpc_minus_random"] > sim["mpc_minus_best_fixed"]
    # and on a state-independent optimum it must NOT beat the best fixed schedule by much
    assert abs(sim["mpc_minus_best_fixed"]) < 0.02


def test_a_zero_noise_simulation_makes_feedback_worthless_by_construction():
    """Guard on the simulator's own premise: with no disturbance there is
    nothing to react to, so MPC and the best fixed schedule must coincide. If
    they differed here, the gap would be an artifact of the planner, not
    control."""

    module = _load()
    starts = {f"y{v:.2f}": float(v) for v in (0.3, 0.5, 0.7)}
    op = {"a": 0.63, "b": 0.072, "c": 0.31, "d": -0.061}
    sim = module.simulate_s3_gap(op, starts, np.array([0.0]), budget=2, start_turn=2, n_turns=20,
                                 late_from=15, n_sims=3, seed=0)
    assert sim["mpc_minus_best_fixed"] == pytest.approx(0.0, abs=1e-9)

    # and the guard has teeth: on a grid too coarse to plan on, the DP loses to
    # the exactly-enumerated schedule and the gap turns spuriously negative
    coarse = module.simulate_s3_gap(op, starts, np.array([0.0]), budget=2, start_turn=2,
                                    n_turns=20, late_from=15, n_sims=3, seed=0, grid_n=21)
    assert coarse["mpc_minus_best_fixed"] < -1e-4


def _count_kernel_from_means(mean: dict) -> np.ndarray:
    """A count kernel with prescribed conditional means, built as two-point
    distributions on the integers straddling each mean."""

    kernel = np.zeros((2, 4, 4))
    for (u, m), value in mean.items():
        lo = int(np.floor(value))
        hi = min(lo + 1, 3)
        frac = value - lo
        kernel[u, m, lo] += 1.0 - frac
        kernel[u, m, hi] += frac
    assert np.allclose(kernel.sum(axis=2), 1.0)
    return kernel


def _affine_kernel(a: float = 0.5, c0: float = 0.5, c1: float = 1.0) -> np.ndarray:
    """NEGATIVE CONTROL for the binary check: `E[m'|m,u] = a m + c_u`, the same
    slope under both actions. A reminder is then worth the same wherever the
    state is, the value function stays linear in `m`, and feedback can buy
    nothing -- the binary-space statement of what killed the scalar arm. If the
    new model class reported headroom here, it would be reporting its own
    arithmetic."""

    return _count_kernel_from_means({(u, m): a * m + (c1 if u else c0)
                                     for u in (0, 1) for m in range(4)})


def _saturating_kernel() -> np.ndarray:
    """POSITIVE CONTROL: a reminder restores violated constraints and can do
    nothing for kept ones, so its value falls as `m` rises. This is the
    mechanism the scalar fit could not represent; the check must be able to
    see it, or "no headroom" on the real kernel would be unfalsifiable."""

    kernel = np.zeros((2, 4, 4))
    for m in range(4):
        kernel[0, m, max(m - 1, 0)] += 0.4
        kernel[0, m, m] += 0.6
        kernel[1, m, 3] += 0.8
        kernel[1, m, m] += 0.2
    return kernel


def test_the_binary_state_reads_the_vector_the_scalar_fit_averaged():
    module = _load()
    rows = _synthetic(a=0.6, b=0.10) + _s1a_rows()
    transitions = module.binary_transitions({"rows": rows}, [])
    assert {t["item_id"] for t in transitions} == {f"tuple_{i}" for i in range(N_ITEMS)}
    # S1a arms carry no excitation and must stay out, exactly as in the scalar fit
    scalar = module.one_step_transitions(module.rows_for_identification({"rows": rows}, []))
    assert len(transitions) == len(scalar)
    # the action credited to a transition is the reminder of the turn it lands in
    by_key = {(r["trajectory_id"], r["turn"]): r for r in rows}
    for t in transitions[:50]:
        assert t["u"] == by_key[(t["trajectory_id"], t["turn"])]["u_remind"]


def test_an_unparsed_constraint_drops_its_pair_from_the_binary_state():
    module = _load()
    rows = _synthetic(a=0.6, b=0.10)
    full = len(module.binary_transitions({"rows": rows}, []))
    for row in rows:
        if row["turn"] == 10 and row["trajectory_id"].endswith("bernoulli__s0"):
            row["followed"] = [True, None, True]
            row["y_graded"] = None
    assert len(module.binary_transitions({"rows": rows}, [])) == full - 2 * N_ITEMS


def test_the_constraint_kernel_recovers_a_planted_state_dependent_gain():
    module = _load()
    rng = np.random.default_rng(0)
    p = {(0, 0): 0.30, (0, 1): 0.80, (1, 0): 0.85, (1, 1): 0.95}  # (b_now, u) -> P(next=1)
    transitions = []
    for item in range(N_ITEMS):
        state = [True, False, True]
        for turn in range(2, 200):
            u = int(rng.integers(0, 2))
            nxt = [bool(rng.random() < p[(int(b), u)]) for b in state]
            transitions.append({"b_now": tuple(state), "b_next": tuple(nxt), "u": u,
                                "turn": turn, "item_id": f"tuple_{item}",
                                "trajectory_id": f"tuple_{item}__bernoulli__s0"})
            state = nxt
    fit = module.fit_constraint_kernel(transitions, seed=0)
    assert fit["cells"]["b=0,u=1"]["p_next_1"] == pytest.approx(0.80, abs=0.03)
    assert fit["cells"]["b=1,u=1"]["p_next_1"] == pytest.approx(0.95, abs=0.03)
    # planted gains: 0.50 when violated, 0.10 when kept
    assert fit["state_dependence_of_the_gain"] == pytest.approx(0.40, abs=0.05)
    assert fit["ci_excludes_zero"]


def test_a_state_independent_action_value_leaves_the_closed_loop_nothing_to_do():
    """The negative control, and the reason the scalar finding was believed:
    when the reminder's worth does not move with the state, the DP picks the
    same turns as the best fixed schedule and the simulated gap is exactly 0
    under common random numbers."""

    module = _load()
    starts = {f"item_{m}": np.eye(4)[m] for m in range(4)}
    sim = module.simulate_binary_s3_gap(_affine_kernel(), starts, budget=3, start_turn=2,
                                        n_turns=20, late_from=15, n_sims=25, seed=0)
    assert sim["mpc_minus_best_fixed"] == pytest.approx(0.0, abs=1e-12)
    sep = module.binary_schedule_separability({"binary_count": _affine_kernel()}, starts,
                                              (1, 2, 3), start_turn=2, n_turns=20, late_from=15)
    assert set(sep["distinct_schedules"]["binary_count"].values()) == {1}


def test_a_saturating_kernel_does_give_the_closed_loop_something_to_do():
    """Positive control. The check must report headroom on a kernel where the
    reminder's value falls as constraints are already kept -- otherwise a null
    on the real data would say nothing about the system."""

    module = _load()
    starts = {f"item_{m}": np.eye(4)[m] for m in range(4)}
    sim = module.simulate_binary_s3_gap(_saturating_kernel(), starts, budget=3, start_turn=2,
                                        n_turns=20, late_from=15, n_sims=50, seed=0)
    assert sim["mpc_minus_best_fixed"] > 0.02
    assert sim["mpc_minus_random"] > sim["mpc_minus_best_fixed"]


def test_the_binary_simulation_pairs_its_draws():
    """Common random numbers, checked the way the scalar version is: a seed
    change must move both policies together, or the gap is the difference of
    two noise draws rather than of two policies."""

    module = _load()
    starts = {f"item_{m}": np.eye(4)[m] for m in range(4)}
    kernel = _saturating_kernel()
    a = module.simulate_binary_s3_gap(kernel, starts, 3, 2, 20, 15, n_sims=200, seed=0)
    b = module.simulate_binary_s3_gap(kernel, starts, 3, 2, 20, 15, n_sims=200, seed=1)
    assert abs(a["mpc_minus_best_fixed"] - b["mpc_minus_best_fixed"]) < 0.02


def test_the_binary_verdict_is_read_against_the_design_mde(tmp_path):
    """The pre-registered rule, as a test: the same simulated gap flips the
    verdict when and only when the MDE it is compared against moves."""

    module = _load()
    readout = json.loads(_readout(_synthetic(a=0.6, b=0.10), tmp_path).read_text())
    common = dict(readout=readout, excluded=[], all_rows=readout["rows"], budgets=(2,),
                  start_turn=2, n_turns=N_TURNS, late_from=15, simulate_budget=2, n_sims=20, seed=0)
    strict = module.binary_state_model(mde=1.0, **common)
    lenient = module.binary_state_model(mde=-1.0, **common)
    assert strict["verdict"] == "DEGENERACY_SURVIVES_THE_MODEL_CLASS"
    assert lenient["verdict"] == "CLOSED_LOOP_HAS_HEADROOM"
    assert strict["s3_gap_simulation"]["mpc_minus_best_fixed"] == pytest.approx(
        lenient["s3_gap_simulation"]["mpc_minus_best_fixed"])


def _memoryless_kernel() -> np.ndarray:
    """NEGATIVE CONTROL for option (b): the next state does not depend on the
    current one, so observing the state tells the controller nothing and
    feedback must buy exactly nothing at equal cost. The threshold objective
    cannot rescue a system with no state -- if it appeared to, the gap would be
    the comparison's own arithmetic rather than control."""

    kernel = np.zeros((2, 4, 4))
    for m in range(4):
        kernel[0, m] = [0.10, 0.25, 0.35, 0.30]
        kernel[1, m] = [0.05, 0.15, 0.35, 0.45]
    return kernel


def test_the_threshold_objective_buys_nothing_on_a_memoryless_system():
    module = _load()
    starts = {f"item_{m}": np.eye(4)[m] for m in range(4)}
    out = module.threshold_objective(_memoryless_kernel(), starts, theta=2, start_turn=2,
                                     n_turns=20, late_from=15, max_k=4,
                                     lambdas=np.linspace(0.0, 0.2, 21), mde=0.05)
    assert out["best_matched_cost_point"]["gap_vs_shared"] == pytest.approx(0.0, abs=1e-9)
    assert out["verdict"] == "DEGENERACY_SURVIVES_THE_OBJECTIVE"


def test_the_threshold_objective_finds_the_headroom_a_saturating_system_has():
    """Positive control. On a kernel where a reminder is worth much more once
    constraints have slipped, waiting to see whether they did must beat any
    schedule fixed in advance at the same expected spend."""

    module = _load()
    starts = {f"item_{m}": np.eye(4)[m] for m in range(4)}
    out = module.threshold_objective(_saturating_kernel(), starts, theta=2, start_turn=2,
                                     n_turns=20, late_from=15, max_k=4,
                                     lambdas=np.linspace(0.0, 0.2, 21), mde=0.02)
    assert out["best_matched_cost_point"]["gap_vs_shared"] > 0.02
    assert out["verdict"] == "WORTH_THE_GPU"


def test_the_closed_loop_is_never_credited_with_spending_more():
    """The guard the matched-cost design exists for: at every frontier point
    the open-loop competitor is priced at the SAME expected number of
    reminders, and the fixed schedules are enumerated exhaustively, so the gap
    can never come from the closed loop simply buying more."""

    module = _load()
    starts = {f"item_{m}": np.eye(4)[m] for m in range(4)}
    out = module.threshold_objective(_saturating_kernel(), starts, theta=2, start_turn=2,
                                     n_turns=20, late_from=15, max_k=4,
                                     lambdas=np.linspace(0.0, 0.2, 21), mde=0.02)
    shared = out["open_loop_frontier"]["shared"]
    # spending more never hurts the open loop, so its frontier is monotone ...
    values = [shared[k]["service_level"] for k in sorted(shared)]
    assert values == sorted(values)
    # ... and the oracle, which picks per item, is never below the shared schedule
    for point in out["frontier"]:
        assert point["open_loop_oracle_at_same_cost"] >= point["open_loop_shared_at_same_cost"] - 1e-12
        assert point["expected_reminders"] <= out["max_reminders_enumerated"]


def test_the_open_loop_envelope_is_mixed_not_rounded_down():
    """A fractional expected cost must be met by the chord between two
    schedules -- a randomised mixture is itself an open-loop policy. Rounding
    down to the cheaper integer schedule would hand the closed loop a gap it
    did not earn."""

    module = _load()
    at, hull = module._upper_envelope([(0.0, 0.0), (1.0, 0.5), (2.0, 0.6), (3.0, 0.6)])
    assert at(0.5) == pytest.approx(0.25)
    assert at(1.5) == pytest.approx(0.55)
    # a dominated point (more cost, no more value) must not extend the hull
    assert [p[0] for p in hull] == [0.0, 1.0, 2.0]


def test_the_threshold_sizing_reports_the_new_primary_not_the_old_one(tmp_path):
    """Changing the objective changes the primary, and the primary's variance
    is what prices the design. The sizing must be computed on the indicator,
    not inherited from mean y -- the 2026-09-10 note ties the arm table, the
    primary and the MDE together for exactly this reason."""

    module = _load()
    pilot = module._sibling_module("analyze_sequor_s1_pilot.py")
    out = module.threshold_sizing(pilot, _sizing_rows()[0], theta=2, late_from=15, n_turns=N_TURNS,
                                  seeds=len(SEEDS), fraction=0.5, seed=0)
    assert out["primary"].startswith("share of turns")
    assert out["status"].startswith("REPORTED, NOT SIGNED")
    assert 0.0 < out["full_dose_contrast"] <= 1.0
    assert out["mde_at_current_design"] > 0
