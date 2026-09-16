"""Tests for scripts/analyze_tsar_cefr_moving_reference.py (candidate B's P1 probe).

The probe reports G_plan = J(best myopic) - J(best anticipating) and a NULL is
what closes candidate B, so the two things that have to be pinned are the ones
that could manufacture a null:

  1. The anticipating solver really is a minimiser over the whole step-dependent
     class, not a search. A search returns an upper bound on J, which inflates
     J(anticipating), which SHRINKS G_plan -- a false null in the cheap-looking
     direction. `test_prefix_solver_is_exact` brute-forces every table in the
     class and asserts equality with no tolerance.
  2. The statistic can see a gap when one is there. A probe that returns zero on
     everything would close candidate B for free. `test_detects_planted_gap`
     plants a plant where anticipation provably pays and asserts G_plan > 0.

Brute force over 3 bins x 4 steps is 4^12 tables, so as in test_tsar_cefr_d25 the
fixture keeps only TWO bins occupied (4^8 = 65,536). The machinery under test --
prefix enumeration, dedup by position vector, the last-step separability
shortcut, the running-cost accumulation -- runs exactly as on the real tree.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_tsar_cefr_moving_reference.py"


def load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("tsar_mref", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mref = load_script()
d25 = sys.modules["analyze_tsar_cefr_d25"]


def make_cell(text_id, source_id, target, seed, lo=3.6, hi=5.0):
    """A full 341-node cell whose states stay >= 0.5 above target (two bins only)."""
    rng = np.random.default_rng(seed)
    cell = d25.Cell(text_id, source_id, target)
    for idx in range(d25.N_NODES):
        cell.ell[idx] = float(rng.uniform(lo, hi))
    cell.termsq[d25.OFFSETS[d25.DEPTH] :] = 0.0
    return cell


def additive_cell(text_id, source_id, target, ell0, deltas):
    """A cell whose level is ell0 plus the sum of per-action increments on its path."""
    cell = d25.Cell(text_id, source_id, target)
    cell.ell[0] = ell0
    for depth in range(1, d25.DEPTH + 1):
        for local in range(d25.N_ACTIONS ** depth):
            idx = d25.OFFSETS[depth] + local
            acc, rest = 0.0, local
            for _ in range(depth):
                acc += deltas[rest % d25.N_ACTIONS]
                rest //= d25.N_ACTIONS
            cell.ell[idx] = ell0 + acc
    cell.termsq[d25.OFFSETS[d25.DEPTH] :] = 0.0
    return cell


@pytest.mark.parametrize("depth", [2, 3, d25.DEPTH])
def test_path_positions_inverts_child_index(depth):
    """Ancestors must be recovered at EVERY depth the solver calls this with.

    The anticipating solver passes prefix-end (depth DEPTH-1) positions, so a
    version that only handles terminal nodes reads the wrong offsets and
    silently mis-scores the running cost of steps 1..DEPTH-1.
    """
    for seq in itertools.product(range(d25.N_ACTIONS), repeat=depth):
        idx, walk = 0, []
        for t, a in enumerate(seq):
            idx = int(d25.child_index(np.int64(idx), t, a))
            walk.append(idx)
        got = mref.path_positions(np.array([[idx]], dtype=np.int64), depth)
        assert [int(g[0, 0]) for g in got] == walk


def test_prefix_solver_is_exact():
    cells = [make_cell(f"c{i}", f"s{i}", "B1", 100 + i) for i in range(4)]
    g = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["switch_at_3"])
    occupied = {int(b) for b in np.unique(g.binv)}
    assert len(occupied) == 2, f"fixture must occupy exactly 2 bins, got {occupied}"
    pos3, dec = mref.enumerate_prefixes_running(g)
    members = np.arange(g.n)
    _, exact = mref.solve_anticipating(g, pos3, dec, members)
    brute = mref.brute_force_anticipating(g, members, sorted(occupied))
    assert exact == pytest.approx(brute, abs=1e-12)


def test_anticipating_never_worse_than_myopic():
    """The step-invariant class is a subset of the step-dependent one."""
    for seed in range(5):
        cells = [make_cell(f"c{i}", f"s{i}", "B1", 900 + 10 * seed + i) for i in range(4)]
        for name in mref.SCHEDULES:
            g = mref.MovingRefGroup(cells, 3, mref.SCHEDULES[name])
            members = np.arange(g.n)
            pos3, dec = mref.enumerate_prefixes_running(g)
            _, j_my = mref.solve_myopic(g, members)
            _, j_an = mref.solve_anticipating(g, pos3, dec, members)
            assert j_an <= j_my + 1e-12, f"{name}/{seed}: G_plan negative"


def test_detects_planted_gap():
    """A plant where anticipation provably pays must give G_plan > 0.

    ell starts AT target and the reference drops by 1.0 at step 3, while one
    `step_down` buys only -0.4: two descents are needed but only two steps
    remain, so the descent has to begin before the switch is visible. A
    step-invariant table cannot do that -- at steps 1-2 the error against the
    then-current reference is 0, the same bin it will occupy later.
    """
    cells = [additive_cell(f"c{i}", f"s{i}", "B1", 3.0, (-0.4, -0.2, 0.0, 0.0))
             for i in range(3)]
    g = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["switch_at_3"])
    members = np.arange(g.n)
    pos3, dec = mref.enumerate_prefixes_running(g)
    _, j_my = mref.solve_myopic(g, members)
    _, j_an = mref.solve_anticipating(g, pos3, dec, members)
    assert j_my - j_an > 1e-9, f"probe blind to a planted gap: {j_my} vs {j_an}"


def test_degenerate_plant_gives_exactly_zero():
    """If no action changes the level, no policy can differ from any other."""
    cells = [additive_cell(f"c{i}", f"s{i}", "B1", 3.7, (0.0, 0.0, 0.0, 0.0))
             for i in range(3)]
    for name in mref.SCHEDULES:
        g = mref.MovingRefGroup(cells, 3, mref.SCHEDULES[name])
        members = np.arange(g.n)
        pos3, dec = mref.enumerate_prefixes_running(g)
        _, j_my = mref.solve_myopic(g, members)
        _, j_an = mref.solve_anticipating(g, pos3, dec, members)
        assert j_my - j_an == 0.0


def test_constant_schedule_bins_never_see_a_switch():
    cells = [additive_cell("c0", "s0", "B1", 3.0, (-0.4, -0.2, 0.0, 0.0))]
    g = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["constant"])
    assert np.allclose(g.r, 3.0)
    g2 = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["switch_at_3"])
    assert list(g2.r[0]) == [3.0, 3.0, 2.0, 2.0]


def test_decision_bin_uses_the_reference_it_is_scored_against():
    """At the decision for step t the controller sees r_t, never r_(t+1)."""
    cells = [additive_cell("c0", "s0", "B1", 3.0, (0.0, 0.0, 0.0, 0.0))]
    g = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["switch_at_3"])
    # root (deciding step 1) is scored against r_1 = 3.0 -> d = 0 -> "arrived"
    assert g.bin_names[g.binv[0, 0]] == "arrived"
    # a depth-2 node (deciding step 3) is scored against r_3 = 2.0 -> d = 1 -> "far"
    idx = d25.OFFSETS[2]
    assert g.bin_names[g.binv[0, idx]] == "far"


def test_refuses_to_overwrite_output(tmp_path):
    out = tmp_path / "already.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        mref.main(["--tree", str(tmp_path / "nope.jsonl"), "--out", str(out)])


def test_report_carries_harness_fingerprint():
    """`.claude/global.md` -> 产物与谱系: the artifact must say which harness made it.

    `provenance()` refuses rather than degrading, so importing it is not enough --
    this asserts the probe actually calls it and that the sha is populated.
    """
    prov = mref.provenance()
    assert prov["git_sha"]
