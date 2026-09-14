"""Tests for scripts/fit_tsar_cefr_operator.py (tsar_cefr Phase 3
identification, plan section 5.1).

Each gate gets its own planted-defect fixture, built independently of the
others (the S3 lesson against one shared end-to-end fixture, restated in
test_tsar_cefr_gpu1_admission.py): a real linear system should pass G-S2-1,
a degenerate/uninformative state should fail it; a real step_down effect
should pass G-S2-2, B == 0 should fail it; G-S2-3 is recorded rather than
gated, so its tests check the diagnostic numbers against hand-designed
matrices with a known spectral radius and a known controllability rank.

The gate-logic tests (G-S2-1/2/3) construct `transitions` dicts directly --
the intermediate representation `build_transitions` produces -- rather than
routing everything through synthetic jsonl rows. This is deliberate: the
statistics inside `run_cv_folds` / `bootstrap_b_step_down` / `gate_s2_3`
only need source_id / turn_next / target_cefr / xi_t / xi_next / u, and
building those directly makes it possible to plant an exact, provable
defect (e.g. a constant xi_t that carries zero information) without fighting
the self-consistency that a real row-chained trajectory imposes (a row's
level_expected is simultaneously this transition's target AND next
transition's history). `build_transitions` itself -- the row-parsing path --
gets its own separate, narrower tests below.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fit_tsar_cefr_operator.py"
sys.path.insert(0, str(ROOT / "src"))


def load_script():
    spec = importlib.util.spec_from_file_location("tsar_op", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


op = load_script()

from persona_drift.modeling.koopman import surrogate_from_arrays, no_extra_features  # noqa: E402


# =============================================================================
# build_transitions: the row-parsing path, its own small fixture
# =============================================================================

def make_row(text_id, source_id, seed, step, action, level_expected, meaning_to_source,
             source_level_expected, target_cefr="A2"):
    return {
        "text_id": text_id, "source_id": source_id, "target_cefr": target_cefr,
        "seed": seed, "step": step, "action": action,
        "level_expected": level_expected, "meaning_to_source": meaning_to_source,
        "source_level_expected": source_level_expected,
    }


def test_build_transitions_lag1_embedding_and_transition_count():
    """A single 3-step trajectory: with lag=1, transitions exist for t=1,2
    (range(1, max_t)), not for t=0 (no history before the source)."""
    rows = [
        make_row("01-a2", "01", 0, 1, "step_down", level_expected=3.0,
                  meaning_to_source=0.9, source_level_expected=4.0),
        make_row("01-a2", "01", 0, 2, "paraphrase", level_expected=2.5,
                  meaning_to_source=0.85, source_level_expected=4.0),
        make_row("01-a2", "01", 0, 3, "copy", level_expected=2.5,
                  meaning_to_source=0.85, source_level_expected=4.0),
    ]
    transitions = op.build_transitions(rows, lag=1)
    assert len(transitions) == 2
    by_t = {tr["t"]: tr for tr in transitions}

    # xi_1 = [ell_1, s_1, ell_0(source), s_0=1.0]
    np.testing.assert_allclose(by_t[1]["xi_t"], [3.0, 0.9, 4.0, 1.0])
    # xi_2 = [ell_2, s_2, ell_1, s_1]
    np.testing.assert_allclose(by_t[1]["xi_next"], [2.5, 0.85, 3.0, 0.9])
    assert by_t[1]["action"] == "paraphrase"  # the action that PRODUCES step 2
    assert by_t[1]["turn_next"] == 2

    np.testing.assert_allclose(by_t[2]["xi_t"], [2.5, 0.85, 3.0, 0.9])
    np.testing.assert_allclose(by_t[2]["xi_next"], [2.5, 0.85, 2.5, 0.85])
    assert by_t[2]["action"] == "copy"
    assert by_t[2]["source_id"] == "01"
    assert by_t[2]["target_cefr"] == "A2"


def test_build_transitions_drops_the_named_exclusion():
    rows = [
        make_row("51-a2", "51", 0, s, "copy", level_expected=3.0, meaning_to_source=0.9,
                  source_level_expected=3.0)
        for s in (1, 2, 3)
    ] + [
        make_row("52-a2", "52", 0, s, "copy", level_expected=3.0, meaning_to_source=0.9,
                  source_level_expected=3.0)
        for s in (1, 2, 3)
    ]
    transitions = op.build_transitions(rows, lag=1)
    assert {tr["text_id"] for tr in transitions} == {"52-a2"}
    assert "51-a2" not in {tr["text_id"] for tr in transitions}


# =============================================================================
# G-S2-1: one-step held-out MSE vs. three trivial nulls
# =============================================================================

def _make_transition(source_id, xi_t, u_vec, action, xi_next, turn_next, target_cefr):
    return {"text_id": f"{source_id}-x", "source_id": source_id, "seed": 0,
            "target_cefr": target_cefr, "t": turn_next - 1, "turn_next": turn_next,
            "xi_t": np.asarray(xi_t, dtype=float), "xi_next": np.asarray(xi_next, dtype=float),
            "u": np.asarray(u_vec, dtype=float), "action": action}


def _linear_system_transitions(n_sources=40, per_source=15, seed=0, noise_sd=1e-3):
    """A REAL linear system: xi_(t+1) = K xi_t + B u_t + c + tiny noise, with
    xi_t drawn from a wide, source-heterogeneous range. Only a model that
    actually reads xi_t can track that heterogeneity -- none of the three
    nulls are given xi_t at all -- so this should clear G-S2-1 with room to
    spare."""
    rng = np.random.default_rng(seed)
    K = np.array([[0.7, 0.05, 0.1, 0.0],
                  [0.02, 0.6, 0.0, 0.05],
                  [1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    B = np.array([[-0.5, -0.25, -0.05],
                  [-0.03, -0.015, -0.02],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    c = np.array([0.6, 0.1, 0.0, 0.0])
    actions = op.ACTIONS + (op.REFERENCE_ACTION,)

    transitions = []
    for i in range(n_sources):
        source_id = f"s{i:03d}"
        for _ in range(per_source):
            xi_t = np.array([rng.uniform(1.0, 6.0), rng.uniform(0.3, 1.0),
                              rng.uniform(1.0, 6.0), rng.uniform(0.3, 1.0)])
            action = actions[rng.integers(0, 4)]
            u = op.action_one_hot(action)
            xi_next = K @ xi_t + B @ u + c + rng.normal(0, noise_sd, size=4)
            turn_next = int(rng.integers(2, 7))
            target = "A2" if rng.random() < 0.5 else "B1"
            transitions.append(_make_transition(source_id, xi_t, u, action, xi_next,
                                                 turn_next, target))
    return transitions


def test_gate_s2_1_passes_on_a_true_linear_system():
    transitions = _linear_system_transitions()
    cv = op.run_cv_folds(transitions)
    gate = op.gate_s2_1(cv)
    assert gate["n_folds_run"] == op.N_FOLDS
    assert gate["n_folds_beating_best_null"] >= op.MIN_FOLDS_PASS
    assert gate["verdict"] == "PASS"
    assert gate["mean_model_mse"] < gate["mean_best_null_mse"]


def _uninformative_state_transitions(n_sources=40, per_source=15, seed=0):
    """PLANTED DEFECT: xi_t is a CONSTANT tuple for every single transition
    (zero variance -- no information a linear map on xi_t could ever
    exploit), while the target is bimodal by `target_cefr` with negligible
    noise. `stateless` (which is handed target_level_rank directly) recovers
    the target almost exactly; a model that only ever sees the same xi_t
    cannot, and is forced toward predicting the average of the two modes --
    a large, systematic error. This is exactly the failure mode G-S2-1
    exists to catch: the dynamics carry no information the label doesn't
    already give away for free."""
    rng = np.random.default_rng(seed)
    actions = op.ACTIONS + (op.REFERENCE_ACTION,)
    const_xi = np.array([3.0, 0.9, 3.0, 0.9])
    transitions = []
    for i in range(n_sources):
        source_id = f"s{i:03d}"
        for _ in range(per_source):
            action = actions[rng.integers(0, 4)]
            u = op.action_one_hot(action)
            turn_next = int(rng.integers(2, 7))
            target = "A2" if rng.random() < 0.5 else "B1"
            ell_next = (1.0 if target == "A2" else 5.0) + rng.normal(0, 1e-3)
            s_next = 0.05 * turn_next + rng.normal(0, 1e-3)
            xi_next = np.array([ell_next, s_next, const_xi[0], const_xi[1]])
            transitions.append(_make_transition(source_id, const_xi, u, action, xi_next,
                                                 turn_next, target))
    return transitions


def test_gate_s2_1_fails_when_the_state_carries_no_information():
    transitions = _uninformative_state_transitions()
    cv = op.run_cv_folds(transitions)
    gate = op.gate_s2_1(cv)
    assert gate["n_folds_beating_best_null"] < op.MIN_FOLDS_PASS
    assert gate["verdict"] == "FAIL"
    assert gate["mean_model_mse"] > gate["mean_best_null_mse"]


# =============================================================================
# G-S2-2: B's step_down column, source_id-clustered bootstrap
# =============================================================================

def _step_down_effect_transitions(n_sources=40, per_source=12, seed=0, step_down_effect=-0.8):
    """A clear, consistent step_down effect on ell_(t+1), small noise, many
    sources -- the bootstrap CI should exclude 0 comfortably."""
    rng = np.random.default_rng(seed)
    actions = op.ACTIONS + (op.REFERENCE_ACTION,)
    transitions = []
    for i in range(n_sources):
        source_id = f"s{i:03d}"
        base = 3.0 + 0.1 * (i % 7)
        for _ in range(per_source):
            xi_t = np.array([base + rng.normal(0, 0.05), 0.9, base, 0.9])
            action = actions[rng.integers(0, 4)]
            u = op.action_one_hot(action)
            effect = step_down_effect if action == "step_down" else 0.0
            ell_next = xi_t[0] + effect + rng.normal(0, 0.02)
            s_next = xi_t[1] - (0.05 if action == "step_down" else 0.0) + rng.normal(0, 0.01)
            xi_next = np.array([ell_next, s_next, xi_t[0], xi_t[1]])
            turn_next = int(rng.integers(2, 7))
            target = "A2" if i % 2 == 0 else "B1"
            transitions.append(_make_transition(source_id, xi_t, u, action, xi_next,
                                                 turn_next, target))
    return transitions


def test_gate_s2_2_passes_with_a_real_step_down_effect():
    transitions = _step_down_effect_transitions()
    boot = op.bootstrap_b_step_down(transitions, draws=2000)
    gate = op.gate_s2_2(boot)
    assert gate["ci_excludes_zero"] is True
    assert gate["verdict"] == "PASS"
    assert gate["point"] < 0  # sanity: step_down should read as lowering ell, not required by the gate


def test_gate_s2_2_fails_when_b_is_exactly_zero():
    """PLANTED DEFECT (as named in the task instructions): zero out the
    action effect entirely. xi_(t+1) depends on xi_t but not at all on the
    action -- B's true step_down column is exactly 0, so the CI must not
    exclude it."""
    transitions = _step_down_effect_transitions(step_down_effect=0.0)
    boot = op.bootstrap_b_step_down(transitions, draws=2000)
    gate = op.gate_s2_2(boot)
    assert gate["ci_excludes_zero"] is False
    assert gate["verdict"] == "FAIL"
    assert gate["ci95"][0] < 0 < gate["ci95"][1]


def test_bootstrap_point_estimate_matches_the_fitted_operator():
    """The guard fit_koopman_sequor_model.py calls `_assert_matches_surrogate`:
    the bootstrap's own normal-equations solve at full sample must equal
    KoopmanSurrogate.fit's B. build_report enforces this via
    `_assert_bootstrap_matches_fit`; this test calls it directly."""
    transitions = _step_down_effect_transitions()
    model = op.fit_operator(transitions)
    boot = op.bootstrap_b_step_down(transitions, draws=100)
    op._assert_bootstrap_matches_fit(model, boot)  # must not raise


# =============================================================================
# G-S2-3: recorded diagnostics (spectral radius / controllability / Gramian)
# =============================================================================

def test_gate_s2_3_reports_a_known_spectral_radius_and_full_rank():
    """A hand-built, well-conditioned system: K diagonal with a known
    largest eigenvalue, B full column rank and reaching every state dim
    within the horizon -- controllability should come back full rank."""
    K = np.diag([0.5, 0.3, 0.2, 0.1])
    B = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 1.0, 1.0]])
    model = surrogate_from_arrays(A=K, B=B, b=[0, 0, 0, 0], C=[[1, 0, 0, 0]],
                                  state_dim=4, extra_features_fn=no_extra_features)
    diag = op.gate_s2_3(model, horizon=4)
    assert diag["spectral_radius"] == pytest.approx(0.5)
    assert diag["controllability_rank"] == 4
    assert diag["full_rank"] is True
    assert diag["gramian_condition"] > 1.0


def test_gate_s2_3_reports_a_known_rank_deficient_system():
    """PLANTED DEFECT for this gate: B's rows are zero for the last two
    state dims, so no input ever reaches them (K is block-diagonal, so K
    cannot route the actuated block into the unactuated one either).
    Controllability rank must come back at most 2, not the full state_dim,
    and G-S2-3 must report `full_rank: False` rather than silently passing."""
    K = np.array([[0.5, 0.1, 0.0, 0.0],
                  [0.1, 0.4, 0.0, 0.0],
                  [0.0, 0.0, 0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.2]])
    B = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    model = surrogate_from_arrays(A=K, B=B, b=[0, 0, 0, 0], C=[[1, 0, 0, 0]],
                                  state_dim=4, extra_features_fn=no_extra_features)
    diag = op.gate_s2_3(model, horizon=4)
    assert diag["controllability_rank"] == 2
    assert diag["full_rank"] is False
    assert diag["criterion"].startswith("recorded only")
