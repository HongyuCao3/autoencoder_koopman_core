"""EK-A analysis (docs/experiments/ergo_fidelity_restoration_plan.md section 4).

These build a synthetic branch arm with a KNOWN causal effect and check the
analyzers recover it -- shape assertions alone would not catch a sign error or
a mis-joined pair, and both would silently decide G-EKA-1/2/3.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from analyze_ergo_branch_pairs import gate_eka1, gate_eka2, load_pairs, record_items  # noqa: E402
from fit_koopman_ergo_branch import _ridge_beta, build_transitions  # noqa: E402

TRUE_GAIN = 0.10          # delta_c at c_prev = 0
TRUE_STATE_SLOPE = -0.20  # reset matters less the closer the agent already is


def _write_arm(tmp_path: pathlib.Path, n_items=40, n_turns=5, seed=0) -> pathlib.Path:
    """closeness follows c_next = 0.5*c_prev + 0.3*shard_frac + effect*u + noise
    with effect(c_prev) = TRUE_GAIN + TRUE_STATE_SLOPE * c_prev."""
    rng = np.random.default_rng(seed)
    d = tmp_path / "arm"
    d.mkdir(parents=True, exist_ok=True)
    base_rows, cf_rows = [], []
    for i in range(n_items):
        item = f"item_{i:03d}"
        tid = f"{item}__seed0"
        c_prev = 0.0  # deterministic start so every planted effect is recoverable from the rows
        for turn in range(1, n_turns + 1):
            sf = turn / n_turns
            common = 0.5 * c_prev + 0.3 * sf + float(rng.normal(0, 0.05))
            effect = TRUE_GAIN + TRUE_STATE_SLOPE * c_prev
            c_plain = common
            c_reset = common + effect
            shared = dict(item_id=item, num_shards=n_turns, turn=turn, seed=0,
                          shard_text=f"shard {turn}", gold_answer="1")
            base_rows.append({**shared, "trajectory_id": tid, "u_reset": 0,
                              "closeness": c_plain, "y_task_success": float(c_plain > 0.6),
                              "agent_message": f"reply {item} t{turn} plain"})
            cf_rows.append({**shared, "trajectory_id": f"{tid}#cf_t{turn}", "u_reset": 1,
                            "closeness": c_reset, "y_task_success": float(c_reset > 0.6),
                            "agent_message": f"reply {item} t{turn} reset",
                            "is_counterfactual": True, "base_trajectory_id": tid,
                            "branch_turn": turn, "base_u_reset": 0})
            c_prev = c_plain
    (d / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base_rows))
    (d / "counterfactual_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cf_rows))
    return d


def test_pairs_are_joined_and_oriented_by_action_not_by_branch(tmp_path):
    pairs = load_pairs(_write_arm(tmp_path))
    assert len(pairs) == 40 * 5
    # The two branches share their noise draw, so delta_c is EXACTLY the planted
    # effect -- a mis-joined pair or a flipped orientation cannot survive this.
    for p in pairs:
        if p["c_prev"] is None:
            continue
        planted = TRUE_GAIN + TRUE_STATE_SLOPE * p["c_prev"]
        assert p["delta_c"] == pytest.approx(planted, abs=1e-9)
    # positive for almost every pair: the planted effect only turns negative once
    # c_prev exceeds TRUE_GAIN / -TRUE_STATE_SLOPE = 0.5, which noise reaches rarely
    assert np.mean([p["c_reset"] > p["c_plain"] for p in pairs]) > 0.98
    # turn 1 has no prior state
    assert all(p["c_prev"] is None for p in pairs if p["turn"] == 1)
    assert all(p["c_prev"] is not None for p in pairs if p["turn"] > 1)


def test_gate_eka1_recovers_the_planted_mean_gain(tmp_path):
    r = gate_eka1(load_pairs(_write_arm(tmp_path)))
    # mean effect = TRUE_GAIN + TRUE_STATE_SLOPE * mean(c_prev); c_prev is small
    assert 0.0 < r["mean_delta_c"] < TRUE_GAIN
    assert r["ci_delta_c"][0] > 0 and r["pass"] is True
    assert r["share_delta_c_positive"] > 0.98
    assert r["n_items"] == 40


def test_gate_eka2_recovers_the_planted_state_slope_including_its_sign(tmp_path):
    r = gate_eka2(load_pairs(_write_arm(tmp_path)))
    assert r["slope_on_c_prev"] == pytest.approx(TRUE_STATE_SLOPE, abs=1e-6)
    assert r["ci_slope"][1] < 0, "a negative planted slope must give a CI strictly below 0"
    assert r["pass"] is True
    assert r["n_pairs_dropped_turn1"] == 40


def test_a_zero_effect_arm_does_not_pass_gate_eka1(tmp_path):
    """The gate has to be able to say no."""
    d = _write_arm(tmp_path)
    cf = [json.loads(l) for l in (d / "counterfactual_pairs.jsonl").open()]
    base = {(json.loads(l)["trajectory_id"], json.loads(l)["turn"]): json.loads(l)
            for l in (d / "trajectories.jsonl").open()}
    for c in cf:  # erase the effect, keep everything else
        c["closeness"] = base[(c["base_trajectory_id"], c["branch_turn"])]["closeness"]
    (d / "counterfactual_pairs.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cf))
    r = gate_eka1(load_pairs(d))
    assert r["mean_delta_c"] == pytest.approx(0.0, abs=1e-12)
    assert r["pass"] is False


def test_a_pair_whose_action_did_not_flip_is_rejected_loudly(tmp_path):
    d = _write_arm(tmp_path)
    cf = [json.loads(l) for l in (d / "counterfactual_pairs.jsonl").open()]
    cf[3]["u_reset"] = 0
    (d / "counterfactual_pairs.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cf))
    with pytest.raises(SystemExit, match="not a flipped same-turn pair"):
        load_pairs(d)


def test_transitions_give_two_rows_per_usable_pair_and_recover_B(tmp_path):
    rows = build_transitions(_write_arm(tmp_path))
    assert len(rows) == 2 * 40 * 4, "turn-1 pairs have no prior state and must be dropped"
    assert {r["u"] for r in rows} == {0.0, 1.0}
    X = np.array([[1.0, r["c_prev"], r["u"], r["shard_frac"]] for r in rows])
    y = np.array([r["c_next"] for r in rows])
    beta = _ridge_beta(X, y, 1e-6)
    # B absorbs the mean of the state-dependent effect; a and g are the planted ones
    assert beta[1] == pytest.approx(0.5, abs=0.08)
    assert beta[3] == pytest.approx(0.3, abs=0.08)
    assert 0.0 < beta[2] < TRUE_GAIN


def test_record_items_separate_echo_from_drift(tmp_path):
    d = _write_arm(tmp_path)
    rec = record_items(load_pairs(d), d)
    assert rec["echo_exact_duplicate_rate"] == 0.0  # synthetic replies all differ
    assert 0.0 < rec["echo_token_jaccard_mean"] < 1.0
    assert set(rec["delta_c_by_turn"]) == {1, 2, 3, 4, 5}


def test_g_eka_3_criterion_is_the_skill_ci_not_the_fold_vote(tmp_path):
    """Calibration on a planted effect returns only 16/20 folds -- with ~45 items
    a 20-fold vote holds ~2 items per fold, so it is a low-powered decision rule.
    The pass criterion is the bootstrap CI on skill; the vote is a record item."""
    import subprocess

    d = _write_arm(tmp_path)
    out = tmp_path / "fit.json"
    root = pathlib.Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts" / "fit_koopman_ergo_branch.py"),
         "--branch-dir", str(d), "--out-path", str(out)],
        check=True, cwd=root, capture_output=True,
    )
    r = json.loads(out.read_text())["g_eka_3"]
    assert r["criterion"] == "item-bootstrap CI on skill excludes 0"
    assert r["ci_skill"][0] > 0 and r["pass"] is True
    assert r["arx_wins_record_only"] < r["fold_vote_reference"], (
        "this fixture is the calibration case: a true effect that the fold vote rejects"
    )


def test_g_eka_3_fails_when_the_state_carries_nothing(tmp_path):
    """c_next independent of c_prev -- the operator must not beat the nulls."""
    rng = np.random.default_rng(1)
    d = tmp_path / "flat"
    d.mkdir(parents=True)
    base, cf = [], []
    for i in range(40):
        item, tid = f"item_{i:03d}", f"item_{i:03d}__seed0"
        for turn in range(1, 6):
            common = 0.3 + float(rng.normal(0, 0.05))
            shared = dict(item_id=item, num_shards=5, turn=turn, seed=0,
                          shard_text=f"s{turn}", gold_answer="1", agent_message=f"r{i}{turn}")
            base.append({**shared, "trajectory_id": tid, "u_reset": 0,
                         "closeness": common, "y_task_success": 0.0})
            cf.append({**shared, "trajectory_id": f"{tid}#cf_t{turn}", "u_reset": 1,
                       "closeness": common, "y_task_success": 0.0,
                       "is_counterfactual": True, "base_trajectory_id": tid,
                       "branch_turn": turn, "base_u_reset": 0})
    (d / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base))
    (d / "counterfactual_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cf))

    import subprocess
    out = tmp_path / "flat_fit.json"
    root = pathlib.Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts" / "fit_koopman_ergo_branch.py"),
         "--branch-dir", str(d), "--out-path", str(out)],
        check=True, cwd=root, capture_output=True,
    )
    r = json.loads(out.read_text())["g_eka_3"]
    assert r["pass"] is False, "a state-free arm must not pass G-EKA-3"
