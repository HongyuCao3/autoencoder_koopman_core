import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_paper_side_experiments_tsar_cefr", REPO / "scripts" / "paper_side_experiments_tsar_cefr.py")
st = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(st)


def test_a_scalar_action_column_raises_instead_of_being_fit():
    """The collapse this column must not survive quietly.

    Folding {step_down, half_step_down, paraphrase} into one number shrinks the
    measured action effect toward zero -- toward agreeing with the mechanism
    claim under test. A silent pass here would read as evidence.
    """
    with pytest.raises(st.ActionEncodingError, match="one-hot"):
        st._assert_action_dofs(np.zeros((50, 1)))
    with pytest.raises(st.ActionEncodingError):
        st._assert_action_dofs(np.zeros((50, 2)))
    st._assert_action_dofs(np.zeros((50, len(st.tsar.ACTIONS))))


def test_state_taps_come_from_the_transition_not_the_readout():
    """Same regression the behavioural side paid for, in this line's layout.

    xi is newest-first in blocks of (ell, s), so tap j is columns [2j, 2j+2)
    and the newest block of z_(t+1) is rows [0, 2). A loading read off C would
    say only which slot holds ell.
    """
    A = np.zeros((4, 4))
    A[0, 0] = 0.8       # next ell loads on ell_t
    A[1, 1] = 0.4       # next s loads on s_t
    A[0, 2] = 0.3       # and on ell_(t-1) -- the non-Markov part
    taps = st._state_taps(A, lag=1)
    assert taps[0] == pytest.approx(np.sqrt(0.8 ** 2 + 0.4 ** 2))
    assert taps[1] == pytest.approx(0.3)


def test_a_markov_operator_reports_no_older_tap_weight():
    A = np.zeros((4, 4))
    A[0, 0], A[1, 1] = 0.9, 0.5
    taps = st._state_taps(A, lag=1)
    assert taps[1] == 0.0


def test_rank_deficient_design_is_undefined_not_zero():
    """P1-a: an unidentifiable action channel must not print a short response.

    `core`'s columns are the live case -- the control is an exact affine
    function of the state -- and the failure mode is that "cannot be measured"
    reads as "measured and small".
    """
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 4))
    V = Z @ rng.normal(size=(4, 3))
    ident = st._identifiability(Z, V)
    assert ident["identified"] is False
    assert ident["undefined_reason"]
    free = st._identifiability(Z, rng.normal(size=(200, 3)))
    assert free["identified"] is True


def test_state_and_action_construction_is_this_line_s_own():
    """Imported, not retyped: the operator has to be the one Table 1 scored."""
    fit = importlib.import_module("fit_tsar_cefr_operator")
    assert st.tsar.action_one_hot is fit.action_one_hot
    assert st.tsar.LAG == fit.LAG
    assert tuple(st.tsar.ACTIONS) == tuple(fit.ACTIONS)
    assert st.tsar.REFERENCE_ACTION == fit.REFERENCE_ACTION


def test_task_level_anchor_carries_the_landed_intervals():
    """A diagnostic must not be quotable without the task-level effect.

    The anchor is copied from the landed column, never recomputed here, so the
    artifact cannot drift away from Table 1.
    """
    if not st.TABLE1_PATH.exists():
        pytest.skip("Table 1's tsar_cefr column is not on disk")
    anchor = st._task_level_anchor()
    landed = json.loads(st.TABLE1_PATH.read_text())
    row3 = anchor["table1_row3_ours_minus_withheld_action"]
    assert row3 == landed["contrasts"]["delay_linear_control_minus_delay_linear_no_control"]
    for key in ("point", "ci_low", "ci_high", "excludes_zero"):
        assert key in row3
    assert anchor["source_git_sha"] == landed["provenance"]["git_sha"]


def _synthetic_rows(rng, *, n_items=8, n_seeds=3, n_steps=5):
    """Trajectories in this line's on-disk shape, with a one-step action.

    ell moves by the action's own amount and nothing else carries over, so a
    correct read has to come back with the response spent in step 1.
    """
    effect = {"step_down": -1.0, "half_step_down": -0.5, "paraphrase": 0.0, "copy": 0.0}
    rows = []
    for item in range(n_items):
        source_level = 4.0
        for seed in range(n_seeds):
            ell = source_level
            for step in range(1, n_steps + 1):
                action = str(rng.choice(["step_down", "half_step_down", "paraphrase", "copy"]))
                ell = ell + effect[action] + 0.01 * rng.normal()
                rows.append({
                    "text_id": f"t{item}", "seed": seed, "step": step,
                    "source_id": f"s{item}", "target_cefr": "B1",
                    "source_level_expected": source_level,
                    "level_expected": ell,
                    "meaning_to_source": 0.9 + 0.01 * rng.normal(),
                    "action": action,
                })
    return rows


def test_a_one_step_action_into_an_integrator_still_reads_as_a_long_response(tmp_path):
    """What the 09-16 response-length statistic cannot tell apart.

    Ground truth here: the action moves ell once and nothing carries it
    further -- "the action cashes out in one step" in the only sense the
    mechanism claim uses. But ell keeps the offset, so ||C A^(k-1) B|| stays
    flat and `response_length` reads the full horizon. The statistic measures
    the effect STILL PRESENT, not the effect ARRIVING, and every column in
    this project has a persistent state by construction. The increments are
    what the claim is about, and they go to zero after step 1.
    """
    rng = np.random.default_rng(0)
    rows_path = tmp_path / "trajectories.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in _synthetic_rows(rng)))

    result = st.run_m1(rows_path)
    assert result["identifiability"]["identified"] is True
    response = result["response"]

    assert response["response_length"] == st.tsar.HORIZON     # reads "long"
    assert response["first_step_share"] < 0.3                 # and "not concentrated"
    increments = response["impulse_increment"]
    assert abs(increments[0]) > 0.5
    assert all(abs(v) < 0.05 * abs(increments[0]) for v in increments[1:])

    per_action = response["per_action_first_step_on_ell"]
    assert per_action["step_down"] < per_action["half_step_down"] < 0.0
    assert result["task_level_anchor"]["table1_row3_ours_minus_withheld_action"]["point"] is not None


# --------------------------------------------------------------------------- A1 / M2

def test_a1_default_cells_match_the_landed_table_1_column():
    """对拍, plan section 10: the A1 panel must not be a second opinion.

    lag 1 and lag 0 at the default window ARE Table 1's rows 6 and 2, so if
    anything in this script's fold split, null, window or bootstrap has
    drifted, these cells stop matching value for value.
    """
    if not st.TABLE1_PATH.exists():
        pytest.skip("Table 1's tsar_cefr column is not on disk")
    landed = json.loads(st.TABLE1_PATH.read_text())
    out = st.run_a1(st.tsar.DEFAULT_ROWS_PATH, matched_window=False)

    for mine, theirs in (("lag_1", "delay_linear_control"), ("lag_0", "markov_linear_control")):
        for field in ("rollout_mse", "skill_h", "n_rows"):
            assert out["rows"][mine][field] == landed["rows"][theirs][field], f"{mine}.{field}"
    mine = out["contrasts"]["lag_1_minus_lag_0"]
    theirs = landed["contrasts"]["delay_linear_control_minus_markov_linear_control"]
    for field in ("point", "ci_low", "ci_high", "excludes_zero"):
        assert mine[field] == theirs[field], field
    assert out["best_null_name"] == landed["best_null_name"]
    assert out["n_scored_rows"] == landed["n_scored_rows"]


def test_item_level_reaching_a_scored_step_raises():
    """The leakage guard signed in plan section 10.

    A leak here does not fail loudly -- it makes the baseline stronger, which
    reads as "the delay window was not carrying dynamics after all".
    """
    trajs = [{"text_id": "a", "seed": 0, "ell": {0: 4.0, 1: 3.5, 2: 3.0}}]
    with pytest.raises(st.LeakageError, match="item level"):
        st._item_levels(trajs, prefix_turn=2, scored_min_turn=2)


def test_item_level_uses_only_the_prefix():
    trajs = [{"text_id": "a", "seed": 0, "ell": {0: 4.0, 1: 3.0, 2: 1.0, 3: 0.0}}]
    levels = st._item_levels(trajs, prefix_turn=1, scored_min_turn=2)
    assert levels["a|0"] == pytest.approx(3.5)


def test_the_action_guard_still_screens_the_base_channel_under_m2_augmentation():
    """M2 appends the item level to the action, so V legitimately gains a
    column -- the guard has to keep checking the action block itself."""
    traj = {
        "text_id": "a", "seed": 0, "source_id": "a", "target_level_rank": 2.0, "max_t": 3,
        "ell": {0: 4.0, 1: 3.5, 2: 3.0, 3: 2.5}, "s": {0: 1.0, 1: 0.9, 2: 0.9, 3: 0.9},
        "u": {1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 1.0, 0.0]),
              3: np.array([0.0, 0.0, 1.0])},
    }
    data = st._dataset([traj], st.tsar.LAG, levels={"a|0": 3.75})
    assert data["V"].shape[1] == len(st.tsar.ACTIONS) + 1

    collapsed = dict(traj, u={k: np.array([float(v.argmax())]) for k, v in traj["u"].items()})
    with pytest.raises(st.ActionEncodingError):
        st._dataset([collapsed], st.tsar.LAG, levels={"a|0": 3.75})
