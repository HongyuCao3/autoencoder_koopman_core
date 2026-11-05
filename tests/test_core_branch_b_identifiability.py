import importlib.util
import pathlib

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "core_branch_b", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "core_branch_b_identifiability.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _collect(cb_true, n_traj=252, turns=10, seed=0):
    """A linear plant collected the way --command-policy branch collects it.

    Main trajectories ask for the target r every turn (the landed protocol).
    Each prompt has its own level a_i, CORRELATED with r -- the cross-prompt
    confound the fork design is meant to remove. One fork per trajectory at
    k in {2, 4, 6} with three uniform asks.
    """
    rng = np.random.default_rng(seed)
    rows, points = [], []
    for i in range(n_traj):
        tid, r = f"t{i}", rng.uniform(0.05, 0.95)
        a = 0.4 * (r - 0.5) + 0.05 * rng.standard_normal()
        k = int(rng.choice((2, 4, 6)))

        def step(y_prev, c):
            return 0.3 * y_prev + cb_true * c + a + 0.02 * rng.standard_normal()

        y = {1: rng.uniform(0, 1)}
        for t in range(2, turns + 1):
            y[t] = step(y[t - 1], r)
        for t in range(1, turns + 1):
            rows.append({"trajectory_id": tid, "turn": t, "normalized_output": y[t], "command_norm": r,
                         "command_policy": "branch"})
        asks = rng.uniform(0, 1, 3)
        points.append({"trajectory_id": tid, "k": k, "target_norm": r,
                       "prefix_state": {"y_k_minus_1": {"norm": y[k - 1]},
                                        "y_k_minus_2": None if k == 2 else {"norm": y[k - 2]}},
                       "branches": [{"command_norm": c, "normalized_output": step(y[k - 1], c)} for c in asks]})
    return rows, points


def test_paired_difference_recovers_known_b_despite_prompt_level_confound():
    rows, points = _collect(0.6)
    me = mod.paired_interval(mod.fork_units(points), np.random.default_rng(0))
    assert abs(me["point"] - 0.6) < 0.02 and me["ci95"][0] > 0.5 and me["ci95"][1] < 0.7
    assert me["n_forks"] == 252 and me["n_siblings"] == 756


def test_error_collection_is_unidentified_and_forks_make_it_identified():
    rows, points = _collect(0.6)
    T = mod.all_transitions(rows, points)
    main = mod.identifiability(mod.subset(T, "main"))
    assert main["rank_deficient"] and not main["identified"] and not main["passes_rank_and_r2"]
    full = mod.identifiability(T)
    assert full["passes_rank_and_r2"] and full["identified"] and max(full["r_on_state_r2"]) < 0.99
    assert (T["src"] == "branch").sum() == 3 * sum(p["k"] != 2 for p in points)


def test_zero_effect_interval_contains_zero():
    _, points = _collect(0.0)
    me = mod.paired_interval(mod.fork_units(points), np.random.default_rng(0))
    assert me["ci95"][0] < 0 < me["ci95"][1]
