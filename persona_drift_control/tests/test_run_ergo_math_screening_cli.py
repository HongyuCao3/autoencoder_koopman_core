"""E4c (docs/experiments/two_task_success_plan.md section 2 E4 / section 13.2 /
section 13.6): CLI-wiring tests for scripts/run_ergo_math_screening.py's
`fixed_t_and_last`/`randsched_t_and_last` CONTROLLER_CHOICES entries and the
new `--koopman-objective`/`--koopman-forced-last-reset`/
`--koopman-pad-short-history` flags.

Imports the script via importlib (mirrors test_state_action_interaction_guard.py's
pattern for a non-package `scripts/*.py` module) and drives
`build_controller_factory(args)` directly with a hand-built `argparse.Namespace`
-- this is the function `main()` was refactored to call, purely so these
branches are testable without `main()`'s GPU-bound `run_ergo_math_screening`
call. `load_ergo_koopman_mpc_controller` itself needs no GPU/torch (pure
numpy), so the `ergo_koopman_mpc` branch is exercised directly too, with a
small on-disk fit report.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_ergo_math_screening.py"
_spec = importlib.util.spec_from_file_location("_rems_cli", _SCRIPT)
_rems = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rems)

# num_shards=4 items whose _excitation_seed(0, entry_id)-derived draw is known
# to differ (computed directly from persona_drift.controller_cli._excitation_seed
# + persona_drift.ergo_controllers.RandSchedTAndLastController against the real
# vendored bank): t=1, t=2, t=3 respectively. Locks 13.2's hard requirement.
_SAME_SHARDS_DIFFERENT_T_ITEMS = ("ergo_GSM8K_1246", "ergo_GSM8K_435", "ergo_GSM8K_40")


def _args(controller: str, **overrides) -> argparse.Namespace:
    """Mirrors parse_args()'s defaults for every field build_controller_factory
    reads, so a test only needs to override what it cares about."""

    base = dict(
        controller=controller,
        reset_mode="overwrite",
        prompt_profile="legacy",
        fixed_schedule_turns=None,
        random_excite_p=None,
        random_schedule_spend_prob=1.0,
        remind_budget=None,
        fixed_t=None,
        koopman_model_path=None,
        koopman_model_key="arx",
        koopman_nu=1,
        koopman_mu=1,
        koopman_horizon=2,
        koopman_repeat_penalty=0.0,
        koopman_y_col="closeness",
        koopman_objective="terminal",
        koopman_forced_last_reset=False,
        koopman_pad_short_history=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _fit_report_path(tmp_path, a=0.9, b_gain=0.2) -> pathlib.Path:
    report_path = tmp_path / "koopman_fit_report_closeness.json"
    report_path.write_text(
        '{"arx": {"A": [[%r, 0.0], [0.0, 1.0]], "B": [[%r], [0.0]], '
        '"b": [0.0, 0.0], "C": [[1.0, 0.0]]}}' % (a, b_gain)
    )
    return report_path


# ---------------------------------------------------------------------------
# CLI default
# ---------------------------------------------------------------------------


def test_koopman_objective_cli_default_is_terminal(monkeypatch, tmp_path):
    # Spec: the pre-registered value is fine as a *CLI* default (visible on
    # the command line/sbatch); only the class's own default must stay "sum".
    report_path = _fit_report_path(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ergo_math_screening.py",
            "--output-dir", str(tmp_path / "out"),
            "--controller", "ergo_koopman_mpc",
            "--koopman-model-path", str(report_path),
        ],
    )
    args = _rems.parse_args()
    assert args.koopman_objective == "terminal"
    assert args.koopman_forced_last_reset is False
    assert args.koopman_pad_short_history is False


# ---------------------------------------------------------------------------
# 13.2: RandSchedTAndLastController seeding through the CLI factory
# ---------------------------------------------------------------------------


def test_randsched_t_and_last_different_items_get_different_t(capsys):
    args = _args("randsched_t_and_last")
    factory = _rems.build_controller_factory(args)
    pairs = [(item_id, factory(0, item_id).t) for item_id in _SAME_SHARDS_DIFFERENT_T_ITEMS]
    print("(item, t) pairs (all num_shards=4, seed=0):", pairs)
    ts = [t for _, t in pairs]
    assert len(set(ts)) > 1, f"all items collapsed onto the same t: {pairs}"


def test_randsched_t_and_last_reproducible_same_seed_same_item():
    args = _args("randsched_t_and_last")
    factory = _rems.build_controller_factory(args)
    first = factory(0, "ergo_GSM8K_1246")
    second = factory(0, "ergo_GSM8K_1246")
    assert first.t == second.t
    assert first.turns == second.turns


# ---------------------------------------------------------------------------
# new controllers via the CLI factory: exactly 2 resets, last turn forced
# ---------------------------------------------------------------------------


def _drive_full_trajectory(controller, num_shards):
    return [controller.next_u_remind(turn, []) for turn in range(1, num_shards + 1)]


@pytest.mark.parametrize("cli_controller", ["fixed_t_and_last", "randsched_t_and_last"])
def test_new_controllers_via_cli_factory_exactly_two_resets_last_turn_forced(cli_controller):
    overrides = {"fixed_t": 2} if cli_controller == "fixed_t_and_last" else {}
    args = _args(cli_controller, **overrides)
    factory = _rems.build_controller_factory(args)
    # ergo_GSM8K_621 has num_shards=6 (t=2 for fixed_t_and_last is non-degenerate).
    controller = factory(0, "ergo_GSM8K_621")
    resets = _drive_full_trajectory(controller, num_shards=6)
    assert sum(resets) == 2
    assert resets[-1] == 1


def test_fixed_t_and_last_requires_fixed_t():
    args = _args("fixed_t_and_last")
    with pytest.raises(ValueError):
        _rems.build_controller_factory(args)


# ---------------------------------------------------------------------------
# --koopman-objective sum vs terminal actually land differently on the
# controller built through the CLI factory (not just string-equality of
# .objective -- the decaying-example scenario from
# test_ergo_koopman_mpc.py::test_sum_and_terminal_objectives_choose_opposite_actions_on_a_decaying_example,
# replayed through a controller built via build_controller_factory).
# ---------------------------------------------------------------------------


def test_koopman_objective_cli_flag_actually_changes_controller_behavior(tmp_path):
    import numpy as np

    report_path = tmp_path / "koopman_fit_report_closeness.json"
    report_path.write_text(
        '{"arx": {"A": [[0.5, 0.0], [0.0, 1.0]], "B": [[1.0], [0.0]], '
        '"b": [0.0, 0.0], "C": [[1.0, 0.0]]}}'
    )
    z0 = np.array([0.0, 0.0])

    sum_args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path, koopman_objective="sum",
        koopman_mu=0, koopman_repeat_penalty=0.0, koopman_horizon=2,
    )
    sum_controller = _rems.build_controller_factory(sum_args)(0)
    assert sum_controller.objective == "sum"
    sum_val0 = sum_controller._simulate(z0, 0, remaining_steps=1, remaining_budget=1)
    sum_val1 = sum_controller._simulate(z0, 1, remaining_steps=1, remaining_budget=1)
    assert sum_val1 > sum_val0  # "sum" prefers spending the reset immediately

    terminal_args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path, koopman_objective="terminal",
        koopman_mu=0, koopman_repeat_penalty=0.0, koopman_horizon=2,
    )
    terminal_controller = _rems.build_controller_factory(terminal_args)(0)
    assert terminal_controller.objective == "terminal"
    terminal_val0 = terminal_controller._simulate(z0, 0, remaining_steps=1, remaining_budget=1)
    terminal_val1 = terminal_controller._simulate(z0, 1, remaining_steps=1, remaining_budget=1)
    assert terminal_val0 > terminal_val1  # "terminal" prefers saving it for the last turn


def test_koopman_forced_last_reset_cli_flag_passes_through(tmp_path):
    report_path = _fit_report_path(tmp_path)
    off_args = _args("ergo_koopman_mpc", koopman_model_path=report_path, koopman_forced_last_reset=False)
    # remind_budget=2 required now (the coordinator's follow-up finding, below):
    # forced_last_reset=True with a k=1 budget silently collapses into fixed_last.
    on_args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path, koopman_forced_last_reset=True, remind_budget=2,
    )
    off_controller = _rems.build_controller_factory(off_args)(0)
    on_controller = _rems.build_controller_factory(on_args)(0)
    assert off_controller.forced_last_reset is False
    assert on_controller.forced_last_reset is True


# ---------------------------------------------------------------------------
# CLI-level guard (coordinator's follow-up finding on this same E4c task):
# forced_last_reset=True at remind_budget<2 collapses the arm into fixed_last
# on every turn (verified: T=6, k=1 -> u=[0,0,0,0,0,1], identical to
# fixed_last) while still carrying the mpc_..._forcedlast_... name -- must be
# stopped at the CLI, not silently accepted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("remind_budget", [None, 1])
def test_forced_last_reset_at_k1_raises(tmp_path, remind_budget):
    report_path = _fit_report_path(tmp_path)
    args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path,
        koopman_forced_last_reset=True, remind_budget=remind_budget,
    )
    with pytest.raises(ValueError, match="remind-budget"):
        _rems.build_controller_factory(args)


def test_forced_last_reset_at_k2_does_not_raise(tmp_path):
    report_path = _fit_report_path(tmp_path)
    args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path,
        koopman_forced_last_reset=True, remind_budget=2,
    )
    controller = _rems.build_controller_factory(args)(0)
    assert controller.forced_last_reset is True
    assert controller.remind_budget == 2


# ---------------------------------------------------------------------------
# C1: pre-registered values visible in the arm name; E1's append-suffix
# wrapper stacks on top.
# ---------------------------------------------------------------------------


def test_mpc_arm_name_includes_preregistered_values(tmp_path):
    report_path = _fit_report_path(tmp_path)
    args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path,
        koopman_objective="terminal", koopman_forced_last_reset=True, koopman_pad_short_history=True,
        remind_budget=2,
    )
    controller = _rems.build_controller_factory(args)(0)
    assert controller.name == "mpc_terminal_forcedlast_pad"


def test_mpc_arm_name_gets_append_suffix_in_append_mode(tmp_path):
    report_path = _fit_report_path(tmp_path)
    args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path,
        koopman_objective="terminal", koopman_pad_short_history=True, reset_mode="append",
    )
    controller = _rems.build_controller_factory(args)(0)
    assert controller.name == "mpc_terminal_pad_append"


# --- prompt_profile (docs/experiments/ergo_fidelity_restoration_plan.md R1) ---


def test_arm_name_gets_up_suffix_under_upstream_profile(tmp_path):
    report_path = _fit_report_path(tmp_path)
    args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path,
        koopman_objective="terminal", koopman_pad_short_history=True,
        prompt_profile="upstream",
    )
    controller = _rems.build_controller_factory(args)(0)
    assert controller.name == "mpc_terminal_pad_up"


def test_up_and_append_suffixes_compose_in_a_fixed_order(tmp_path):
    report_path = _fit_report_path(tmp_path)
    args = _args(
        "ergo_koopman_mpc", koopman_model_path=report_path,
        koopman_objective="terminal", koopman_pad_short_history=True,
        prompt_profile="upstream", reset_mode="append",
    )
    factory = _rems.build_controller_factory(args)
    controller = factory(0)
    assert controller.name == "mpc_terminal_pad_up_append"
    # The mpc controller is a singleton returned by reference, so a second
    # factory call must not re-suffix either wrapper (the double-suffix guard
    # that the "_append" wrapper already carries, now needed on "_up" too).
    assert factory(0).name == "mpc_terminal_pad_up_append"


def test_legacy_profile_leaves_arm_names_untouched():
    args = _args("constant_remind", prompt_profile="legacy", reset_mode="overwrite")
    assert _rems.build_controller_factory(args)(0).name == "constant_remind"
