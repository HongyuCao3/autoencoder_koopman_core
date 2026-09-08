import numpy as np
import pytest

from persona_drift.ergo_koopman_mpc import ErgoKoopmanMPCController, load_ergo_koopman_mpc_controller, shard_frac
from persona_drift.modeling.dataset import ReducedStateConfig
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features


def _ergo_rows(*turn_y_u_shards):
    """turn_y_u_shards: (turn, y, u_reset, num_shards) tuples, oldest first.
    Writes the same value under both `closeness` (the current default
    `y_col`) and `y_task_success` (kept for tests that exercise column
    configurability explicitly) -- these mechanism tests don't care which
    real-world readout the number represents."""

    return [
        {"turn": turn, "closeness": y, "y_task_success": y, "u_reset": u, "num_shards": num_shards}
        for turn, y, u, num_shards in turn_y_u_shards
    ]


def _known_surrogate_with_aux(a=0.9, g=0.2, aux_self=1.0, c=0.0, cross=0.0):
    """state z = [y, shard_frac] (nu=1, mu=0, one aux col): y_(t+1) = a*y_t
    + cross*aux_t + g*u_t + c, shard_frac_(t+1) = aux_self*shard_frac_t
    (readout only reads y, C=[1, 0])."""

    model = KoopmanSurrogate(extra_features_fn=no_extra_features)
    model.state_dim = 2
    model.A = np.array([[a, cross], [0.0, aux_self]])
    model.B = np.array([[g], [0.0]])
    model.b = np.array([c, 0.0])
    model.C = np.array([[1.0, 0.0]])
    return model


def test_shard_frac_reads_turn_over_num_shards():
    assert shard_frac({"turn": 3, "num_shards": 6}) == 0.5


def test_current_state_appends_aux_after_y_and_v_hist():
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=1, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=1
    )
    history = _ergo_rows((1, 0.2, 0, 4), (2, 0.5, 1, 4))
    z = controller._current_state(history)
    assert z is not None
    # nu=1 -> y_hist=[0.5] (turn2's own y); mu=1, shift=0 (no contemporaneous_v) ->
    # v_hist=[u at turn1]=[0.0] (turn2's own u_reset is excluded, same lag convention
    # as control.KoopmanMPCController._current_state); aux -> shard_frac(turn=2,
    # num_shards=4) = 0.5, evaluated at the most recent row (turn2), matching
    # build_reduced_state_pairs's aux_now = series[t].
    assert list(z) == [0.5, 0.0, 0.5]


def test_uses_configurable_y_and_u_columns_not_persona_drift_names():
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(surrogate=surrogate, state_config=config, aux_fns=(shard_frac,))
    # rows only have closeness/y_task_success/u_reset, never y_probe/u_remind
    history = _ergo_rows((1, 0.3, 0, 4))
    assert controller.next_u_remind(2, history) in (0, 1)


def test_reminds_when_it_improves_predicted_success_and_penalty_is_zero():
    surrogate = _known_surrogate_with_aux(a=0.9, g=0.2, c=0.0)
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=2, repeat_penalty=0.0
    )
    history = _ergo_rows((1, 0.5, 0, 4))
    assert controller.next_u_remind(2, history) == 1


def test_stops_when_repeat_penalty_dominates():
    # objective="sum" pinned explicitly: under the new default
    # objective="terminal" (E4a), a non-final turn's own repeat_penalty is
    # discarded from the returned total (B4(a) -- terminal ignores every
    # intermediate step's own `value`), so this test's premise ("a huge
    # repeat_penalty stops the controller from reminding") only holds under
    # the additive "sum" objective it was written against.
    surrogate = _known_surrogate_with_aux(a=0.9, g=0.2, c=0.0)
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=2, repeat_penalty=100.0,
        objective="sum",
    )
    history = _ergo_rows((1, 0.5, 0, 4))
    assert controller.next_u_remind(2, history) == 0


def test_falls_back_to_zero_with_insufficient_history():
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=1, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(surrogate=surrogate, state_config=config, aux_fns=(shard_frac,))
    assert controller.next_u_remind(1, []) == 0


def test_load_ergo_koopman_mpc_controller_builds_a_working_controller(tmp_path):
    report_path = tmp_path / "koopman_fit_report_closeness.json"
    report_path.write_text(
        '{"arx": {"A": [[0.9, 0.0], [0.0, 1.0]], "B": [[0.2], [0.0]], '
        '"b": [0.0, 0.0], "C": [[1.0, 0.0]]}}'
    )
    controller = load_ergo_koopman_mpc_controller(
        model_path=report_path, model_key="arx", nu=1, mu=0, horizon=2, repeat_penalty=0.0
    )
    assert controller.y_col == "closeness"
    assert controller.u_col == "u_reset"
    assert controller.state_config.aux_cols == ("shard_frac",)
    assert controller.state_config.contemporaneous_v is True
    history = _ergo_rows((1, 0.5, 0, 4))
    assert controller.next_u_remind(2, history) in (0, 1)


# ---------------------------------------------------------------------------
# F3 (signal_resolution_plan.md section 4.1): shard_frac truth-override
# during MPC lookahead, and per-trajectory dynamic episode_length.
# ---------------------------------------------------------------------------


def test_truth_override_sets_aux_to_exact_turn_over_num_shards_for_3_lookahead_steps():
    # C reads the aux dimension directly (C=[0, 1]) so the returned "value"
    # at remaining_steps=0 is exactly z_next's (truth-overridden) aux
    # component -- isolates the override formula from the y/action logic.
    model = KoopmanSurrogate(extra_features_fn=no_extra_features)
    model.state_dim = 2
    model.A = np.array([[0.9, 0.0], [0.0, 1.0]])
    model.B = np.array([[0.0], [0.0]])
    model.b = np.array([0.0, 0.0])
    model.C = np.array([[0.0, 1.0]])
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=model, state_config=config, aux_fns=(shard_frac,), horizon=1, repeat_penalty=0.0
    )
    controller._lookahead_turn = 5
    controller._num_shards = 10
    z0 = np.array([0.3, 0.7])  # aux placeholder -- irrelevant, always overridden on the way out
    for k in range(3):
        value = controller._simulate(z0, action=0, remaining_steps=0, remaining_budget=None, lookahead_offset=k)
        assert value == pytest.approx((5 + k) / 10)


def test_missing_num_shards_falls_back_to_parent_behavior_without_raising():
    # Empty history: `next_u_remind` never reaches `history[-1]` (real
    # trajectory rows always carry `num_shards`, so this is the one
    # realistic path to `_num_shards is None` -- see next_u_remind's `if
    # history else None`), and the parent's own insufficient-history
    # fallback (`_current_state` returns None) kicks in before `shard_frac`
    # would ever be evaluated.
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=2
    )
    action = controller.next_u_remind(1, [])
    assert action == 0
    assert controller._num_shards is None
    assert controller.n_missing_num_shards == 1


def test_truth_override_propagates_to_y_via_cross_term_and_matches_hand_computation():
    # y_(t+1) = 0.5*y_t + 0.4*aux_t + 0*u (B=0 so action never matters here,
    # isolating the override's effect from the max-over-actions branch).
    # objective="sum" pinned explicitly: this test's assertions are additive
    # totals (depth-0 value + leaf value), the "sum" objective's semantics,
    # not the new default "terminal" (E4a) which would return only the leaf.
    model = KoopmanSurrogate(extra_features_fn=no_extra_features)
    model.state_dim = 2
    model.A = np.array([[0.5, 0.4], [0.0, 1.0]])
    model.B = np.array([[0.0], [0.0]])
    model.b = np.array([0.0, 0.0])
    model.C = np.array([[1.0, 0.0]])
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=model, state_config=config, aux_fns=(shard_frac,), horizon=2, repeat_penalty=0.0,
        objective="sum",
    )
    controller._lookahead_turn = 2
    controller._num_shards = 4
    z0 = np.array([1.0, 0.9])  # z0 itself is never overridden, only surrogate.step's outputs are

    total = controller._simulate(z0, action=0, remaining_steps=1, remaining_budget=None, lookahead_offset=0)
    # depth 0: y1 = 0.5*1.0 + 0.4*0.9 = 0.86; aux overridden to (2+0)/4 = 0.5
    # depth 1: y2 = 0.5*0.86 + 0.4*0.5 = 0.63 (action-independent since B=0)
    assert total == pytest.approx(0.86 + 0.63)

    # Without truth override (num_shards unknown), aux free-runs via A[1,1]=1.0
    # instead, giving a different (and in general wrong) total.
    controller._num_shards = None
    total_naive = controller._simulate(z0, action=0, remaining_steps=1, remaining_budget=None, lookahead_offset=0)
    # depth 0: y1 = 0.86 (unchanged, doesn't depend on aux path); aux stays 0.9
    # depth 1: y2 = 0.5*0.86 + 0.4*0.9 = 0.79
    assert total_naive == pytest.approx(0.86 + 0.79)
    assert total != pytest.approx(total_naive)


def test_remaining_budget_reads_u_reset_not_u_remind():
    # g=0.2 > 0 so, unbudgeted, the controller always wants to reset -- this
    # isolates whether the budget is actually being enforced from whether
    # the controller would ever choose to reset at all.
    surrogate = _known_surrogate_with_aux(a=0.9, g=0.2, c=0.0)
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=2, repeat_penalty=0.0,
        remind_budget=1,
    )
    # Turn 1 already spent the single reset (u_reset=1 recorded in history).
    # Before the _remaining_budget fix, this read the nonexistent "u_remind"
    # key, always saw 0 spent, and never stopped resetting -- reproduced by
    # 15645632 (mpc arm) spending on 548/664 rows against a budget of 1/trajectory.
    history = _ergo_rows((1, 0.5, 1, 4))
    assert controller._remaining_budget(history) == 0
    assert controller.next_u_remind(2, history) == 0


def test_episode_length_is_taken_from_each_trajectorys_own_num_shards():
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=5)
    assert controller.episode_length is None  # unset before any decision

    controller.next_u_remind(2, _ergo_rows((1, 0.5, 0, 4)))
    assert controller.episode_length == 4

    controller.next_u_remind(2, _ergo_rows((1, 0.5, 0, 9)))
    assert controller.episode_length == 9


# ---------------------------------------------------------------------------
# E4a (docs/experiments/two_task_success_plan.md section 2 E4, and section
# 12.2 B3/B4, section 12.3 C1): objective="terminal"/"sum", forced_last_reset,
# pad_short_history=True. Item 1 of the E4 spec (horizon clipped to
# num_shards / episode_length set per-turn / _remaining_budget reads u_col /
# shard_frac truth-override) was confirmed already done (F0/F3) by reading
# control.py:287-299 and this module's existing _remaining_budget/
# next_u_remind/_simulate -- not re-implemented here.
# ---------------------------------------------------------------------------


def test_sum_and_terminal_objectives_choose_opposite_actions_on_a_decaying_example():
    # y decays each step (A[0,0]=0.5) and a reset adds a flat +1 (B[0]=1,
    # repeat_penalty=0). With only one reset to spend (remaining_budget=1)
    # and one more turn to go (remaining_steps=1): spending it now gives a
    # transient bump that half decays away by the final step; saving it for
    # the final step keeps the full +1. "sum" rewards the transient bump
    # (spend now); "terminal" only reads the final step (save it). Verified
    # numerically: sum totals are (action=0: 1.0, action=1: 1.5); terminal
    # totals are (action=0: 1.0, action=1: 0.5).
    model = KoopmanSurrogate(extra_features_fn=no_extra_features)
    model.state_dim = 2
    model.A = np.array([[0.5, 0.0], [0.0, 1.0]])
    model.B = np.array([[1.0], [0.0]])
    model.b = np.array([0.0, 0.0])
    model.C = np.array([[1.0, 0.0]])
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    z0 = np.array([0.0, 0.0])

    sum_controller = ErgoKoopmanMPCController(
        surrogate=model, state_config=config, aux_fns=(shard_frac,), horizon=2,
        repeat_penalty=0.0, objective="sum",
    )
    sum_val0 = sum_controller._simulate(z0, 0, remaining_steps=1, remaining_budget=1)
    sum_val1 = sum_controller._simulate(z0, 1, remaining_steps=1, remaining_budget=1)
    assert sum_val1 > sum_val0  # "sum" prefers spending the reset immediately

    terminal_controller = ErgoKoopmanMPCController(
        surrogate=model, state_config=config, aux_fns=(shard_frac,), horizon=2,
        repeat_penalty=0.0, objective="terminal",
    )
    terminal_val0 = terminal_controller._simulate(z0, 0, remaining_steps=1, remaining_budget=1)
    terminal_val1 = terminal_controller._simulate(z0, 1, remaining_steps=1, remaining_budget=1)
    assert terminal_val0 > terminal_val1  # "terminal" prefers saving it for the last turn


def test_invalid_objective_raises_value_error():
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    with pytest.raises(ValueError):
        ErgoKoopmanMPCController(
            surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), objective="bogus"
        )


def test_terminal_objective_breaks_exact_ties_toward_action_zero():
    # g=0 (B=0): a reset has zero effect on y, and repeat_penalty=0, so both
    # actions produce exactly the same terminal value -- an exact tie. The
    # parent's `if value > best_value` (control.py) is strict and evaluates
    # action=0 first, so it must win; this is the B4(b) predetermined
    # tie-break rule, inherited unchanged, not to be replaced with `>=` or
    # jittered.
    model = KoopmanSurrogate(extra_features_fn=no_extra_features)
    model.state_dim = 2
    model.A = np.array([[0.9, 0.0], [0.0, 1.0]])
    model.B = np.array([[0.0], [0.0]])
    model.b = np.array([0.0, 0.0])
    model.C = np.array([[1.0, 0.0]])
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=model, state_config=config, aux_fns=(shard_frac,), horizon=2,
        repeat_penalty=0.0, objective="terminal",
    )
    history = _ergo_rows((1, 0.5, 0, 4))
    assert controller.next_u_remind(2, history) == 0


def test_planning_steps_clip_to_the_trajectorys_own_num_shards():
    # Item 1 of the E4 spec, confirmed already done by F0/F3 (control.py's
    # `_planning_steps` clips to `episode_length - turn + 1`, and this
    # module's `next_u_remind` sets `episode_length` from `num_shards` every
    # turn) -- this test locks that confirmed behavior, it does not
    # reimplement it. horizon=12 (E4's `--koopman-horizon 12`) is clipped
    # down to 2 for a 3-shard item decided at turn 2.
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=12)
    controller.next_u_remind(2, _ergo_rows((1, 0.5, 0, 3)))
    assert controller.episode_length == 3
    assert controller._planning_steps(2) == min(12, 3 - 2 + 1)
    assert controller._planning_steps(2) == 2


def _run_forced_last_trajectory(controller, num_shards, y=0.5):
    """Steps the controller through every turn of a single num_shards-turn
    trajectory the way a real run loop would: each turn's own decision is
    appended to `history` before deciding the next turn."""

    history: list[dict] = []
    for turn in range(1, num_shards + 1):
        action = controller.next_u_remind(turn, history)
        history.append(
            {"turn": turn, "closeness": y, "y_task_success": y, "u_reset": action, "num_shards": num_shards}
        )
    return history


def _forced_last_surrogate():
    # a=0.9, g=0.2>0: unbudgeted, the controller always wants to reset, so
    # any turn where it does NOT reset is because forced_last_reset's
    # budget reservation (B3), not because resetting stopped helping.
    return _known_surrogate_with_aux(a=0.9, g=0.2, c=0.0)


def test_forced_last_reset_spends_exactly_k_resets_per_trajectory():
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    for k in (1, 2, 3):
        controller = ErgoKoopmanMPCController(
            surrogate=_forced_last_surrogate(), state_config=config, aux_fns=(shard_frac,), horizon=12,
            repeat_penalty=0.0, objective="terminal", remind_budget=k, forced_last_reset=True,
        )
        history = _run_forced_last_trajectory(controller, num_shards=5)
        assert sum(row["u_reset"] for row in history) == k


def test_forced_last_reset_always_resets_on_the_last_turn():
    # Isolated from the "exactly k" count above (failure mode 4 / B3): a
    # uniform "budget - 1 every turn" implementation would let the parent's
    # `remaining_budget <= 0` early return (control.py) silently swallow the
    # forced last-turn reset while still spending k elsewhere, so the total
    # count alone would not catch it -- this checks the last row specifically.
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=_forced_last_surrogate(), state_config=config, aux_fns=(shard_frac,), horizon=12,
        repeat_penalty=0.0, objective="terminal", remind_budget=1, forced_last_reset=True,
    )
    history = _run_forced_last_trajectory(controller, num_shards=5)
    assert history[-1]["turn"] == 5
    assert history[-1]["u_reset"] == 1


def test_forced_last_reset_with_k_equals_2_spends_exactly_one_reset_mid_trajectory():
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=_forced_last_surrogate(), state_config=config, aux_fns=(shard_frac,), horizon=12,
        repeat_penalty=0.0, objective="terminal", remind_budget=2, forced_last_reset=True,
    )
    history = _run_forced_last_trajectory(controller, num_shards=5)
    mid_resets = [row["u_reset"] for row in history[:-1]]
    assert sum(mid_resets) == 1
    assert history[-1]["u_reset"] == 1
    assert sum(row["u_reset"] for row in history) == 2


def test_pad_short_history_true_makes_turn_2_a_real_decision():
    # C1: contemporaneous_v=True, mu=2 -- without padding the earliest real
    # decision is turn 3 (control.py's pad_short_history docstring); with
    # pad_short_history=True (this module's new default) it's turn 2.
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=2, contemporaneous_v=True, aux_cols=("shard_frac",))
    history = _ergo_rows((1, 0.5, 0, 4))

    padded = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=2, pad_short_history=True
    )
    assert padded._current_state(history) is not None

    unpadded = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=2, pad_short_history=False
    )
    assert unpadded._current_state(history) is None


def test_pad_short_history_defaults_to_true():
    # C1: "现在定死" -- the default lives at the dataclass/construction layer
    # (no CLI flag), so any caller that does not pass pad_short_history gets
    # True, including load_ergo_koopman_mpc_controller's factory.
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(surrogate=surrogate, state_config=config, aux_fns=(shard_frac,))
    assert controller.pad_short_history is True
