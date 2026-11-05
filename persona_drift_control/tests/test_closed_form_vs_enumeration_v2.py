import importlib.util
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_closed_form_vs_enumeration_v2", REPO / "scripts" / "closed_form_vs_enumeration_v2.py")
v2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v2)
v1 = v2.v1

K = np.array([[0.8, -0.1, 0.05, -0.2], [0.0, 0.9, 0.0, 0.02], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
B = np.array([[-0.07, -0.05, -0.03], [-0.01, -0.02, 0.01], [0, 0, 0], [0, 0, 0]], dtype=float)
C = np.array([0.7, 0.1, 0, 0], dtype=float)
CANDS = v1.candidates(["step_down", "half_step_down", "paraphrase"])
NAMES = list(CANDS)


def test_fill_a_is_the_exact_discrete_minimum_of_the_held_command_cost():
    """No relaxation: fill A must equal brute force over the four held sequences."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        xi = list(rng.uniform(0.5, 1.5, 4) * [4, 1, 4, 1])
        star, beta, lam = rng.uniform(1, 5), rng.uniform(0, 3), rng.uniform(0, 0.3)
        costs = v1.sequence_costs(K, B, C, xi, star, beta, lam, CANDS)
        held = {a: costs[(a,) * v1.HORIZON] for a in NAMES}
        assert v2.discrete_first(costs, NAMES, v2.FILLS["A_hold_first_command"]) == min(held, key=held.get)


def test_fill_b_scores_first_action_then_copy():
    costs = {("copy",) * 2: 3.0, ("step_down", "copy"): 1.0, ("step_down", "step_down"): 0.0}
    assert v2.discrete_first(costs, ["copy", "step_down"], v2.FILLS["B_first_then_copy"], horizon=2) == "step_down"
    costs[("step_down", "copy")] = 3.0
    assert v2.discrete_first(costs, ["copy", "step_down"], v2.FILLS["B_first_then_copy"], horizon=2) == "copy"


def test_discrete_ties_break_in_candidate_order():
    costs = {("copy", "copy"): 1.0, ("step_down", "step_down"): 1.0}
    assert v2.discrete_first(costs, ["copy", "step_down"], v2.FILLS["A_hold_first_command"], horizon=2) == "copy"
