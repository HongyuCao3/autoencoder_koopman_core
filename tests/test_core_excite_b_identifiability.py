import importlib.util
import pathlib

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "core_excite_b", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "core_excite_b_identifiability.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _rows(b_true, n_traj=60, turns=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_traj):
        y = rng.uniform(0, 1)
        for t in range(1, turns + 1):
            c = rng.uniform(0, 1)
            if t > 1:
                y = 0.5 * y + b_true * c + 0.05 + 0.01 * rng.standard_normal()
            rows.append({"trajectory_id": f"t{i}", "turn": t, "normalized_output": y, "command_norm": c})
    return rows


def test_transitions_align_command_with_next_reply():
    rows = _rows(0.3, n_traj=1)
    (Z, c, Zn), = mod.transitions(rows, lag=3).values()
    by_turn = {r["turn"]: r for r in rows}
    assert Z.shape == (6, 4) and Zn.shape == (6, 4)  # turns 4..9 -> next 5..10
    assert c[0] == by_turn[5]["command_norm"]
    assert Z[0][0] == by_turn[4]["normalized_output"] and Z[0][3] == by_turn[1]["normalized_output"]
    assert Zn[0][0] == by_turn[5]["normalized_output"]


def test_recovers_known_command_effect_and_flags_zero_effect():
    units = list(mod.transitions(_rows(0.3), lag=3).values())
    me = mod.main_element(units, np.random.default_rng(0))
    assert abs(me["point"] - 0.3) < 0.02 and me["ci95"][0] > 0
    assert mod.identifiability(units)["rank"] == 6
    null = mod.main_element(list(mod.transitions(_rows(0.0), lag=3).values()), np.random.default_rng(0))
    assert null["ci95"][0] < 0 < null["ci95"][1]
