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
                        "followed": [True, True, True], "echo_jaccard_prev": None,
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
                        "seed": s, "y_graded": 0.5, "followed": [True, True, True],
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
                        "seed": s, "y_graded": float(y),
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
