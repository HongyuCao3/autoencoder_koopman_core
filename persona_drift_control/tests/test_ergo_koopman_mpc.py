import numpy as np

from persona_drift.ergo_koopman_mpc import ErgoKoopmanMPCController, load_ergo_koopman_mpc_controller, shard_frac
from persona_drift.modeling.dataset import ReducedStateConfig
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features


def _ergo_rows(*turn_y_u_shards):
    """turn_y_u_shards: (turn, y_task_success, u_reset, num_shards) tuples, oldest first."""
    return [
        {"turn": turn, "y_task_success": y, "u_reset": u, "num_shards": num_shards}
        for turn, y, u, num_shards in turn_y_u_shards
    ]


def _known_surrogate_with_aux(a=0.9, g=0.2, aux_self=1.0, c=0.0):
    """state z = [y, shard_frac] (nu=1, mu=0, one aux col): y_(t+1) = a*y_t
    + g*u_t + c, shard_frac_(t+1) = aux_self*shard_frac_t (readout only
    reads y, C=[1, 0])."""

    model = KoopmanSurrogate(extra_features_fn=no_extra_features)
    model.state_dim = 2
    model.A = np.array([[a, 0.0], [0.0, aux_self]])
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
    # rows only have y_task_success/u_reset, never y_probe/u_remind
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
    surrogate = _known_surrogate_with_aux(a=0.9, g=0.2, c=0.0)
    config = ReducedStateConfig(nu=1, mu=0, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(
        surrogate=surrogate, state_config=config, aux_fns=(shard_frac,), horizon=2, repeat_penalty=100.0
    )
    history = _ergo_rows((1, 0.5, 0, 4))
    assert controller.next_u_remind(2, history) == 0


def test_falls_back_to_zero_with_insufficient_history():
    surrogate = _known_surrogate_with_aux()
    config = ReducedStateConfig(nu=1, mu=1, aux_cols=("shard_frac",))
    controller = ErgoKoopmanMPCController(surrogate=surrogate, state_config=config, aux_fns=(shard_frac,))
    assert controller.next_u_remind(1, []) == 0


def test_load_ergo_koopman_mpc_controller_builds_a_working_controller(tmp_path):
    report_path = tmp_path / "koopman_fit_report.json"
    report_path.write_text(
        '{"arx": {"A": [[0.9, 0.0], [0.0, 1.0]], "B": [[0.2], [0.0]], '
        '"b": [0.0, 0.0], "C": [[1.0, 0.0]]}}'
    )
    controller = load_ergo_koopman_mpc_controller(
        model_path=report_path, model_key="arx", nu=1, mu=0, horizon=2, repeat_penalty=0.0
    )
    assert controller.y_col == "y_task_success"
    assert controller.u_col == "u_reset"
    assert controller.state_config.aux_cols == ("shard_frac",)
    assert controller.state_config.contemporaneous_v is True
    history = _ergo_rows((1, 0.5, 0, 4))
    assert controller.next_u_remind(2, history) in (0, 1)
