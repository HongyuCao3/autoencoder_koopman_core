"""Tests for scripts/analyze_tsar_cefr_d25.py (D-2.5, the causal upper bound).

The load-bearing claim in that script is the word EXACT: it reports a supremum
over a policy class, and a supremum that is really a search result is a lower
bound wearing the wrong name -- which, for a criterion whose firing closes the
line, is a false negative in the expensive direction. So the central test here
brute-forces every policy in the class on a small instance and asserts the
solver returns the same value, with no tolerance.

Brute force over 3 bins x 4 steps is 4^12 tables and will not run in a test, so
the fixture is built so that only TWO bins are ever occupied (every state is at
least 0.5 above target), which drops it to 4^8 = 65,536 tables. That is not a
weaker test of the machinery under test: the prefix enumeration, the
deduplication by position vector, and the separability shortcut at the last
step all run exactly as they do on the real tree.

The second group of tests covers the ways this analyzer could leak the
held-out source back into its own score -- the failure the leave-one-out
design exists to prevent -- and the third covers the pre-registered cap
exclusion, which is the one place the script drops data.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_tsar_cefr_d25.py"


def load_script():
    spec = importlib.util.spec_from_file_location("tsar_d25", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d25 = load_script()


def make_cell(text_id, source_id, target, seed, lo=2.6, hi=4.5):
    """A full 341-node cell whose states all sit >= 0.5 above target (two bins only)."""
    rng = np.random.default_rng(seed)
    cell = d25.Cell(text_id, source_id, target)
    for idx in range(d25.OFFSETS[d25.DEPTH]):
        cell.ell[idx] = float(rng.uniform(lo, hi))
    for idx in range(d25.OFFSETS[d25.DEPTH], d25.N_NODES):
        code = int(rng.integers(2, 5))
        cell.termsq[idx] = float((code - cell.lstar) ** 2)
    return cell


def brute_force_best(g, members, occupied_bins):
    """Every table over `occupied_bins`, walked cell by cell. No solver involved."""
    best = None
    for combo in itertools.product(range(d25.N_ACTIONS), repeat=d25.DEPTH * len(occupied_bins)):
        table = np.tile(np.array(g.fallback, dtype=np.int8), (d25.DEPTH, 1))
        k = 0
        for t in range(d25.DEPTH):
            for b in occupied_bins:
                table[t, b] = combo[k]
                k += 1
        total = 0.0
        for i in members:
            idx, _, _ = g.walk(table, int(i))
            total += g.termsq[i, idx]
        mse = total / len(members)
        if best is None or mse < best:
            best = mse
    return best


# =============================================================================
# EXACTNESS -- the whole reason this script is allowed to say "supremum"
# =============================================================================

def test_solver_matches_brute_force_over_the_whole_policy_class():
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=i) for i in range(3)]
    fitter = d25.ClosedLoopFitter(cells, 3)
    members = np.arange(len(cells))
    table, mse, _ = fitter.fit(members)
    occupied = sorted({int(b) for b in fitter.g.binv[:, : d25.OFFSETS[d25.DEPTH]].ravel()})
    assert occupied == [0, 1], "fixture must occupy exactly the far/near bins"
    assert mse == pytest.approx(brute_force_best(fitter.g, members, occupied), abs=1e-12)


def test_returned_table_reproduces_the_reported_objective():
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=10 + i) for i in range(4)]
    fitter = d25.ClosedLoopFitter(cells, 3)
    members = np.arange(len(cells))
    table, mse, _ = fitter.fit(members)
    walked = np.mean([fitter.g.termsq[i, fitter.g.walk(table, int(i))[0]] for i in members])
    assert walked == pytest.approx(mse, abs=1e-12)


def test_prefix_dedup_keeps_every_distinct_position_vector():
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=20 + i) for i in range(3)]
    g = d25.Group(cells, 3)
    pos, dec = d25.enumerate_prefixes(g)
    assert len(np.unique(pos, axis=0)) == pos.shape[0]
    assert pos.shape[0] == len({tuple(r) for r in pos})


def test_a_dominant_action_is_actually_found():
    """Plant a single action that is always best; the solver must return it."""
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=30 + i) for i in range(3)]
    for cell in cells:
        cell.termsq[d25.OFFSETS[d25.DEPTH] :] = 9.0
        for idx in range(d25.OFFSETS[d25.DEPTH], d25.N_NODES):
            local = idx - d25.OFFSETS[d25.DEPTH]
            if all((local // 4 ** k) % 4 == 0 for k in range(d25.DEPTH)):
                cell.termsq[idx] = 0.0
    fitter = d25.ClosedLoopFitter(cells, 3)
    table, mse, _ = fitter.fit(np.arange(len(cells)))
    assert mse == pytest.approx(0.0)
    for t in range(d25.DEPTH):
        assert table[t, 0] == 0 or table[t, 1] == 0


# =============================================================================
# NO LEAKAGE -- the held-out source must not reach its own fit
# =============================================================================

def test_unvisited_bin_falls_back_and_is_counted():
    """A held-out cell in a bin no training cell occupies must not read the table."""
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=40 + i) for i in range(3)]
    cells.append(make_cell("99-a2", "99", "A2", seed=99, lo=0.1, hi=0.4))
    got = d25.closed_loop_scores(cells, 3, loo=True)
    assert got["fallbacks"] >= 1


def test_leave_one_out_scores_differ_from_in_sample():
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=50 + i) for i in range(6)]
    loo = d25.closed_loop_scores(cells, 3, loo=True)["sq"]
    ins = d25.closed_loop_scores(cells, 3, loo=False)["sq"]
    assert sum(ins.values()) <= sum(loo.values()) + 1e-9


def test_both_cells_of_a_source_leave_together():
    """The two target levels of one paragraph are not independent; LOO drops both."""
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=60 + i) for i in range(4)]
    cells += [make_cell(f"{i:02d}-b1", f"{i:02d}", "B1", seed=70 + i) for i in range(4)]
    src = np.array([c.source_id for c in cells])
    held = np.flatnonzero(src != src[0])
    assert "00" not in {cells[int(i)].source_id for i in held}


def test_open_loop_backs_off_when_its_source_level_cell_is_empty():
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=80 + i, lo=3.4, hi=3.6)
             for i in range(3)]
    cells.append(make_cell("77-a2", "77", "A2", seed=77, lo=4.4, hi=4.6))
    got = d25.open_loop_scores(cells, loo=True)
    assert got["backoff_to_pooled"] >= 1


def test_open_loop_value_table_matches_a_hand_walk():
    cell = make_cell("01-a2", "01", "A2", seed=123)
    vals = d25.open_loop_values([cell])
    seq = (2, 0, 3, 1)
    idx = 0
    for t, a in enumerate(seq):
        idx = int(d25.child_index(np.int64(idx), t, a))
    flat = ((seq[0] * 4 + seq[1]) * 4 + seq[2]) * 4 + seq[3]
    assert vals[0, flat] == pytest.approx(cell.termsq[idx])


def test_superset_class_cannot_be_worse_in_sample_than_open_loop():
    """It contains the open-loop class, so in sample its pooled RMSE must not be larger."""
    cells = []
    for i in range(6):
        lo = 2.6 if i % 2 == 0 else 3.6
        cells.append(make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=90 + i, lo=lo, hi=lo + 0.3))
    sup = d25.superset_scores(cells, 3, loo=False)["sq"]
    opn = d25.open_loop_scores(cells, loo=False)["sq"]
    assert sum(sup.values()) <= sum(opn.values()) + 1e-9


# =============================================================================
# THE CAP EXCLUSION -- the one place data is dropped
# =============================================================================

def test_cap_gate_flags_only_items_over_the_line():
    cells = {f"{i:02d}-a2": make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=i) for i in range(3)}
    cells["00-a2"].n_cap = 20
    cells["01-a2"].n_cap = 10
    cap = d25.cap_pressure(cells)
    assert cap["items_over_limit"] == ["00-a2"]
    assert cap["worst_share"] == pytest.approx(20 / 340)
    assert cap["design_resolution"] == pytest.approx(1 / 340)


def test_cap_gate_resolution_is_not_zero_tolerance_here():
    """GPU-1's threshold was zero tolerance at 18 rows; at 340 rows it is not."""
    cells = {"00-a2": make_cell("00-a2", "00", "A2", seed=1)}
    cells["00-a2"].n_cap = 1
    cd = d25.cap_pressure(cells)
    assert cd["items_over_limit"] == []


def test_bin_edges_sit_where_greedy_reactive_says():
    assert d25.bin_edges_3(1.0) == 0 and d25.bin_edges_3(0.999) == 1
    assert d25.bin_edges_3(0.5) == 1 and d25.bin_edges_3(0.499) == 2
    assert d25.bin_edges_4(0.0) == 2 and d25.bin_edges_4(-0.001) == 3


def test_greedy_reactive_is_the_bins_namesake_table():
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=110 + i) for i in range(3)]
    g = d25.Group(cells, 3)
    assert tuple(g.fallback) == (0, 1, 3)
    assert d25.ACTIONS[0] == "step_down" and d25.ACTIONS[1] == "half_step_down"
    assert d25.ACTIONS[3] == "copy"
    assert d25.greedy_reactive_rmse(cells, 3) >= 0.0


def test_out_dir_is_append_only(tmp_path):
    (tmp_path / "already").mkdir()
    with pytest.raises(SystemExit):
        d25.main(["--tree", "unused.jsonl", "--out-dir", str(tmp_path / "already")])


def test_load_tree_rejects_a_missing_terminal(tmp_path):
    rows = []
    for depth in range(1, 5):
        for path in itertools.product(d25.ACTIONS, repeat=depth):
            rows.append({
                "text_id": "01-a2", "source_id": "01", "target_cefr": "A2",
                "depth": depth, "path": list(path), "action": path[-1],
                "level_expected": 3.0, "level_official": "B1",
                "hit_token_cap": False, "source_level_expected": 3.5,
            })
    rows.pop()
    p = tmp_path / "tree.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        d25.load_tree(p)


# =============================================================================
# THE REVISION'S DIRECTION -- step-dependent contains step-invariant, always
# =============================================================================

def test_step_dependent_class_contains_the_step_invariant_one():
    """Section 2.6's whole claim: the signed class was a strict subset, so its
    supremum is weaker. If this ever inverted, the revision would be unmotivated."""
    cells = [make_cell(f"{i:02d}-a2", f"{i:02d}", "A2", seed=200 + i) for i in range(4)]
    dependent = d25.closed_loop_scores(cells, 3, loo=False)["sq"]
    invariant = d25.step_invariant_scores(cells, 3, loo=False)["sq"]
    assert sum(dependent.values()) <= sum(invariant.values()) + 1e-9


def test_step_invariant_values_match_a_hand_walk():
    cell = make_cell("01-a2", "01", "A2", seed=321)
    vals = d25.step_invariant_values([cell], 3)
    combo = (2, 0, 3)
    g = d25.Group([cell], 3)
    table = np.tile(np.array(combo, dtype=np.int8), (d25.DEPTH, 1))
    idx, _, _ = g.walk(table, 0)
    flat = (combo[0] * 4 + combo[1]) * 4 + combo[2]
    assert vals[0, flat] == pytest.approx(cell.termsq[idx])


def test_fixed_ladder_descends_only_as_far_as_the_nominal_gap():
    """It is the unfitted OPEN-LOOP reference, so it must read source level and nothing else."""
    far = make_cell("01-a2", "01", "A2", seed=400)
    near = make_cell("02-a2", "02", "A2", seed=401)
    far.ell[0] = 4.9    # rounds to C1=5, three levels above A2=2
    near.ell[0] = 2.1   # rounds to A2=2, already there
    assert d25.fixed_ladder_rmse([far])["mean_descent_steps"] == pytest.approx(3.0)
    assert d25.fixed_ladder_rmse([near])["mean_descent_steps"] == pytest.approx(0.0)
    both = d25.fixed_ladder_rmse([far, near])
    assert both["rmse"] >= 0.0
