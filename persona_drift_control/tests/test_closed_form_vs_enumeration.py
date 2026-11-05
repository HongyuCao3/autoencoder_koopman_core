import importlib.util
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_closed_form_vs_enumeration", REPO / "scripts" / "closed_form_vs_enumeration.py")
cf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cf)

K = np.array([[0.8, -0.1, 0.05, -0.2], [0.0, 0.9, 0.0, 0.02], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
B = np.array([[-0.07, -0.05, -0.03], [-0.01, -0.02, 0.01], [0, 0, 0], [0, 0, 0]], dtype=float)
C = np.array([0.7, 0.1, 0, 0], dtype=float)
CANDS = cf.candidates(["step_down", "half_step_down", "paraphrase"])


def test_closed_form_minimises_the_constant_command_quadratic():
    """c* must beat every point of a grid under the same held-command objective."""
    xi, star, beta, lam = [3.4, 1.0, 3.4, 1.0], 2.0, 1.8, 0.1
    c_star, _ = cf.closed_form(K, B, C, xi, star, beta, lam, CANDS)

    def J(u):
        state, total = np.asarray(xi), 0.0
        for _ in range(cf.HORIZON):
            state = K @ state + B @ u + C
            total += (state[0] - star) ** 2 + beta * (1 - state[1]) + lam * u.sum()
        return total

    grid = [np.array(g) for g in np.random.default_rng(0).uniform(-3, 3, size=(2000, 3))]
    assert all(J(c_star) <= J(g) + 1e-9 for g in grid)


def test_enumeration_breaks_ties_in_action_order_like_the_deployed_controller():
    costs = {("copy", "copy"): 1.0, ("step_down", "copy"): 1.0, ("paraphrase", "copy"): 2.0}
    assert cf.enumerate_first(costs) == "copy"


def test_decisions_rebuild_the_state_and_infer_the_copy_stop():
    rows = [{"arm": "koopman_mpc", "text_id": "x", "seed": 0, "step": 0, "level_expected": 3.0,
             "meaning_to_source": 1.0, "target_cefr": "A2", "action": "start"},
            {"arm": "koopman_mpc", "text_id": "x", "seed": 0, "step": 1, "level_expected": 2.5,
             "meaning_to_source": 0.9, "target_cefr": "A2", "action": "half_step_down"},
            {"arm": "one_shot", "text_id": "x", "seed": 0, "step": 0, "level_expected": 9.0,
             "meaning_to_source": 1.0, "target_cefr": "A2", "action": "start"}]
    out = cf.decisions(rows, max_steps=6)
    assert out == [([3.0, 1.0, 3.0, 1.0], "A2", "half_step_down"),
                   ([2.5, 0.9, 3.0, 1.0], "A2", "copy")]
    assert cf.decisions(rows, max_steps=1) == out[:1]
