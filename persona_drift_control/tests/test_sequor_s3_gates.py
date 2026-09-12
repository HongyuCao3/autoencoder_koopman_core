"""Tests for scripts/analyze_sequor_s3_gates.py.

Every admission check gets a planted defect it must catch, and the estimand
gets a planted effect it must recover to the digit plus a planted null it must
report as zero. The rule behind all of them: a gate that passes silently makes
the failure it exists to catch invisible in the report.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_sequor_s3_gates.py"

ARMS = ["koopman_mpc", "greedy_targeted", "best_fixed_schedule"]
N_ITEMS = 10
N_TURNS = 20
LATE_FROM = 15
SEEDS = [0, 1, 2]
K = 3


def _load():
    spec = importlib.util.spec_from_file_location("_s3_gates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load()


@pytest.fixture(scope="module")
def pilot(module):
    return module._sibling_module("analyze_sequor_s1_pilot.py")


def _y(arm: str, item: int, turn: int, seed: int, effect: dict[str, float]) -> float:
    """A readout with item-to-item spread, a downward turn ramp, per-seed
    wobble, and a planted late-window gain per arm. Quantised to the k+1 values
    `y` can actually take, because a gate reading a continuous stand-in would
    never meet the ceiling it is built to detect."""

    base = 0.05 + 0.06 * item + 0.05 * ((turn + seed) % 3) - 0.004 * turn
    if turn >= LATE_FROM:
        base += effect.get(arm, 0.0)
    return min(1.0, max(0.0, round(base * K) / K))


def _rows(effect: dict[str, float] | None = None, arms=ARMS) -> list[dict]:
    effect = effect or {}
    rows = []
    for arm in arms:
        for item in range(N_ITEMS):
            for seed in SEEDS:
                for turn in range(1, N_TURNS + 1):
                    y = _y(arm, item, turn, seed, effect)
                    followed = [j < round(y * K) for j in range(K)]
                    named = [] if turn == 1 else ([0] if arm != "best_fixed_schedule"
                                                  else ([0, 1, 2] if turn % 7 == 0 else []))
                    if arm != "best_fixed_schedule" and named and followed[0]:
                        named = []  # a closed loop only names what its judge called broken
                    rows.append({
                        "trajectory_id": f"item{item}__{arm}__s{seed}", "item_id": f"item{item}",
                        "turn": turn, "branch": arm, "seed": seed, "u_remind": int(bool(named)),
                        "constraints": [f"c{j}" for j in range(K)],
                        "followed": followed, "y_graded": y,
                        "followed_inloop": followed, "y_inloop": y,
                        "state_before_action": [bool(b) for b in followed],
                        "action_named": named, "n_named": len(named),
                        "inserted_tokens": 21 * len(named), "budget": 400,
                        "tokens_left_after": 400,
                    })
    return rows


def _arm_report(rows, arms=ARMS, blind=0.0, overspent=None) -> dict:
    return {
        "arms": list(arms), "seeds": SEEDS, "n_turns": N_TURNS, "n_items": N_ITEMS,
        "token_cap_by_item": {f"item{i}": {"token_cap_share": 0.0} for i in range(N_ITEMS)},
        "cap_criterion": 0.05, "token_cap_share": 0.0, "max_new_tokens": 2048,
        "gold_coverage": {"n_constraints_judged": 150, "n_constraints_in_calibration_set": 50,
                          "share_outside_calibration_set": 0.667},
        "budget_accounting": {
            arm: {"share_of_budget_spent": 0.9, "n_actions": 10, "n_targeted_actions": 8,
                  "n_blanket_actions": 2, "blind_turn_share": blind,
                  "overspent_items": (overspent or {}).get(arm, [])}
            for arm in arms},
    }


def _run_config() -> dict:
    return {"budget_share": 0.15}


def test_a_planted_late_window_gain_comes_back_exactly(module, pilot):
    rows = _rows({"koopman_mpc": 1 / 3})
    block = module.contrast(pilot, rows, "koopman_mpc", "best_fixed_schedule",
                            range(LATE_FROM, N_TURNS + 1), seed=0)
    assert block["n_items"] == N_ITEMS and block["n_seeds"] == len(SEEDS)
    assert block["contrast"] > 0
    assert block["ci"][0] > 0
    plain = module.contrast(pilot, _rows(), "koopman_mpc", "best_fixed_schedule",
                            range(LATE_FROM, N_TURNS + 1), seed=0)
    # Quantisation to the k+1 values `y` can take moves the recovered gain off the
    # planted one by less than one readout step.
    assert block["contrast"] - plain["contrast"] == pytest.approx(1 / 3, abs=1 / K / 2)


def test_identical_arms_give_exactly_zero(module, pilot):
    """The negative control: the two arms are the same rows under two labels."""

    block = module.contrast(pilot, _rows(), "koopman_mpc", "greedy_targeted",
                            range(LATE_FROM, N_TURNS + 1), seed=0)
    assert block["contrast"] == 0.0
    assert not block["ci_excludes_zero"]
    assert not block["resolved"]


def test_the_estimand_never_reads_the_in_loop_judge(module, pilot):
    """`y_inloop` drove the controller, so it is a selection signal. Corrupting
    it must not move a single reported number."""

    rows = _rows({"koopman_mpc": 1 / 3})
    before = module.contrast(pilot, rows, "koopman_mpc", "best_fixed_schedule",
                             range(LATE_FROM, N_TURNS + 1), seed=0)["contrast"]
    for row in rows:
        row["y_inloop"] = 0.0
        row["followed_inloop"] = [False] * K
    after = module.contrast(pilot, rows, "koopman_mpc", "best_fixed_schedule",
                            range(LATE_FROM, N_TURNS + 1), seed=0)["contrast"]
    assert before == after


def test_admission_passes_on_a_well_formed_arm(module):
    rows = _rows()
    gate = module.gate_g_s3(_arm_report(rows), _run_config(), rows, ARMS, N_TURNS)
    assert gate["verdict"] == "PASS", gate["checks"]


def test_admission_catches_an_open_loop_arm_that_read_the_state(module):
    """The check that could not exist before this arm did. An open-loop arm has
    no way to name the broken constraints; a row where it named a proper subset
    means the runner leaked the state into a schedule, and the comparison would
    be between two closed loops."""

    rows = _rows()
    leaked = next(r for r in rows if r["branch"] == "best_fixed_schedule" and r["turn"] > 1)
    leaked["action_named"] = [1]
    leaked["n_named"] = 1
    gate = module.gate_g_s3(_arm_report(rows), _run_config(), rows, ARMS, N_TURNS)
    assert not gate["checks"]["open_loop_arms_never_targeted"]
    assert gate["open_loop_arms_that_targeted"] == ["best_fixed_schedule"]
    assert gate["verdict"] == "FAIL"


def test_admission_catches_an_action_at_turn_one(module):
    rows = _rows()
    first = next(r for r in rows if r["turn"] == 1)
    first["action_named"], first["n_named"] = [0, 1, 2], 3
    gate = module.gate_g_s3(_arm_report(rows), _run_config(), rows, ARMS, N_TURNS)
    assert not gate["checks"]["turn_1_action_free_in_every_arm"]


def test_admission_catches_a_closed_loop_target_that_was_not_broken(module):
    """A controller naming a constraint its own judge called satisfied is not
    the policy the arm claims to have run."""

    rows = _rows()
    row = next(r for r in rows if r["branch"] == "koopman_mpc" and r["turn"] > 1)
    row["state_before_action"] = [True, True, True]
    row["action_named"], row["n_named"] = [0], 1
    gate = module.gate_g_s3(_arm_report(rows), _run_config(), rows, ARMS, N_TURNS)
    assert not gate["checks"]["closed_loop_targets_were_broken_constraints"]


def test_admission_catches_an_overspent_budget(module):
    rows = _rows()
    report = _arm_report(rows, overspent={"koopman_mpc": [["item0", 0]]})
    gate = module.gate_g_s3(report, _run_config(), rows, ARMS, N_TURNS)
    assert not gate["checks"]["no_arm_overspent_its_budget"]


def test_admission_catches_a_controller_that_was_blind_too_often(module):
    """5% unparsed was measured in advance (15739196); 3x that is an
    instrument fault, not the known discount."""

    rows = _rows()
    assert module.gate_g_s3(_arm_report(rows, blind=0.05), _run_config(), rows, ARMS,
                            N_TURNS)["checks"]["blind_turn_share_under_ceiling"]
    assert not module.gate_g_s3(_arm_report(rows, blind=0.30), _run_config(), rows, ARMS,
                                N_TURNS)["checks"]["blind_turn_share_under_ceiling"]


def test_admission_catches_a_dead_readout(module):
    """If every trajectory in an arm sits at the same `y` on some turn, that
    turn carries no information and the arm cannot be compared on it."""

    rows = _rows()
    for row in rows:
        if row["branch"] == "koopman_mpc" and row["turn"] == 18:
            row["y_graded"] = 1.0
    gate = module.gate_g_s3(_arm_report(rows), _run_config(), rows, ARMS, N_TURNS)
    assert not gate["checks"]["spread_at_every_turn_per_arm"]
    assert "18" in gate["turns_failing_spread_by_arm"]["koopman_mpc"]


def test_admission_catches_missing_rows(module):
    rows = _rows()[:-1]
    gate = module.gate_g_s3(_arm_report(rows), _run_config(), rows, ARMS, N_TURNS)
    assert not gate["checks"]["row_count_matches_design"]


def _block(point, ci, mde=0.0617):
    return {"contrast": point, "ci": list(ci), "ci_excludes_zero": ci[0] * ci[1] > 0,
            "mde_at_80pct_this_arm": mde, "abs_effect_over_mde": abs(point) / mde,
            "resolved": ci[0] * ci[1] > 0 and abs(point) >= mde}


def test_a_win_under_the_mde_is_undecidable_not_a_win(module):
    """The K3 shape: significant at ~1.96 sigma but under the design's own
    80%-power MDE is a point estimate likely inflated by the noise that made it
    detectable."""

    ruling = module.verdict(_block(0.03, (0.004, 0.056)), _block(0.02, (-0.01, 0.05)), None)
    assert ruling["verdict"] == "UNDECIDABLE"


def test_a_resolved_win_is_positive_and_names_what_the_operator_earned(module):
    both = module.verdict(_block(0.09, (0.04, 0.14)), _block(0.08, (0.03, 0.13)), None)
    assert both["verdict"] == "RQ3_POSITIVE"
    assert "the operator earned it" in both["reason"]

    only_primary = module.verdict(_block(0.09, (0.04, 0.14)), _block(0.01, (-0.03, 0.05)), None)
    assert only_primary["verdict"] == "RQ3_POSITIVE"
    assert "cannot say the FITTED OPERATOR earned it" in only_primary["reason"]


def test_a_resolved_loss_closes_the_arm(module):
    ruling = module.verdict(_block(-0.09, (-0.14, -0.04)), _block(-0.05, (-0.09, -0.01)), None)
    assert ruling["verdict"] == "RQ3_NEGATIVE_RESOLVED"
    assert ruling["closes_the_line"] is True


def test_undecidable_reads_as_a_clean_negative_only_when_the_arms_beat_doing_nothing(module):
    """The plan's signed clause. 'The closed loop adds nothing over the best
    fixed schedule' is a publishable result only if the arms did something at
    all; if they did not beat zero_control either, the run says nothing."""

    beat = module.verdict(_block(0.004, (-0.03, 0.04)), _block(0.00, (-0.04, 0.04)),
                          _block(0.15, (0.10, 0.20)))
    assert beat["verdict"] == "UNDECIDABLE" and beat["arms_beat_zero_control"]
    assert "CLEAN NEGATIVE" in beat["consequence"]

    flat = module.verdict(_block(0.004, (-0.03, 0.04)), _block(0.00, (-0.04, 0.04)),
                          _block(0.01, (-0.04, 0.06)))
    assert not flat["arms_beat_zero_control"]
    assert "says nothing about the closed loop" in flat["consequence"]


def test_the_undecidable_reason_names_the_prediction_made_before_submission(module):
    ruling = module.verdict(_block(0.03, (-0.01, 0.07)), _block(0.02, (-0.02, 0.06)), None)
    assert "0.86x MDE" in ruling["reason"]
    assert "not about the closed loop" in ruling["reason"]


def test_inloop_agreement_is_a_diagnostic_computed_on_both_verdicts(module):
    rows = _rows()
    assert module.inloop_vs_independent(rows)["per_constraint_agreement"] == 1.0
    for row in rows:
        row["followed_inloop"] = [not b for b in row["followed"]]
    assert module.inloop_vs_independent(rows)["per_constraint_agreement"] == 0.0
    for row in rows:
        row["followed_inloop"] = None
    assert module.inloop_vs_independent(rows)["n_rows_both_parsed"] == 0


def test_end_to_end_through_main_on_a_synthetic_arm(module, tmp_path, monkeypatch, capsys):
    """The wiring, not the statistics: readout loading, the join that pulls
    action fields back out of trajectories.jsonl, the cap guard, the
    zero_control merge across two arm directories, and the written report.

    It exists because the last pilot's rows sat unjudged for a day with no
    analyzer, and an analyzer that crashes on first contact with real rows is
    the same delay wearing a different hat.
    """

    import json

    arm_dir = tmp_path / "sequor_s3_arm"
    arm_dir.mkdir()
    rows = _rows({"koopman_mpc": 1 / 3})
    (arm_dir / "arm_report.json").write_text(json.dumps(_arm_report(rows)))
    (arm_dir / "run_config.json").write_text(json.dumps(_run_config()))
    with (arm_dir / "trajectories.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    readout = arm_dir / "readout_independent.json"
    readout.write_text(json.dumps({
        "judge_kind": "independent", "mode": "canonical", "arm_dir": str(arm_dir),
        "agent_model": "agent", "judge_model": "judge", "n_rows_unusable": 0,
        "provenance": {"git_sha": "0" * 40}, "rows": rows}))

    zero_rows = [dict(r, branch="zero_control", trajectory_id=r["trajectory_id"] + "__z",
                      y_graded=max(0.0, r["y_graded"] - 1 / 3))
                 for r in rows if r["branch"] == "koopman_mpc" and r["item_id"] != "item9"]
    zero = tmp_path / "zero.json"
    zero.write_text(json.dumps({
        "judge_kind": "independent", "mode": "canonical", "arm_dir": str(tmp_path),
        "agent_model": "agent", "judge_model": "judge", "rows": zero_rows}))

    out = tmp_path / "s3_gates_report.json"
    monkeypatch.setattr("sys.argv", [
        "analyze_sequor_s3_gates.py", "--independent-readout", str(readout),
        "--zero-control-readout", str(zero), "--out-path", str(out),
        "--late-from", str(LATE_FROM)])
    module.main()

    report = json.loads(out.read_text())
    assert report["g_s3"]["verdict"] == "PASS"
    assert report["primary"]["treated"] == "koopman_mpc"
    assert report["primary"]["control"] == "best_fixed_schedule"
    assert report["primary"]["contrast"] > 0
    assert report["secondary"]["koopman_mpc_minus_zero_control"]["n_shared_items"] == N_ITEMS - 1
    assert report["secondary"]["koopman_mpc_minus_zero_control"]["rng_paired"] is False
    assert set(report["arm_levels"]) == set(ARMS)
    assert report["inloop_vs_independent"]["per_constraint_agreement"] == 1.0
    assert report["predicted_before_submission"]["koopman_mpc_minus_best_fixed_schedule"] == 0.0531
    assert "0.67" in report["caveat"] or "67%" in report["caveat"]
    assert report["verdict"]["verdict"] in ("RQ3_POSITIVE", "UNDECIDABLE")
    assert "VERDICT" in capsys.readouterr().out


def test_main_refuses_to_overwrite_an_existing_report(module, tmp_path, monkeypatch):
    out = tmp_path / "existing.json"
    out.write_text("{}")
    monkeypatch.setattr("sys.argv", [
        "analyze_sequor_s3_gates.py", "--independent-readout", str(tmp_path / "nope.json"),
        "--out-path", str(out)])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        module.main()
