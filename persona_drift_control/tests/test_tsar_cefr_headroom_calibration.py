"""Tests for scripts/analyze_tsar_cefr_headroom_calibration.py (candidate D).

Candidate D measures the BIAS of the leave-one-out headroom estimator by
planting a known-size step-dependent preference into the cost. Three things can
silently invalidate that measurement, and each has a test here:

  1. The planting leaks into the published path. `run_schedule` grew an optional
     `plant` argument, and the probe's own verdict must not move by a bit.
     `test_plant_none_is_byte_identical` pins that.
  2. The planted preference is not actually step-dependent. If a step-invariant
     table could follow `pistar` everywhere, the plant would add cost to both
     classes equally and the "true headroom" column would stay flat while delta
     grew, which reads as a biased estimator when it is a broken plant.
     `test_pistar_is_not_step_invariant` and `test_true_headroom_grows_with_delta`
     pin that.
  3. The plant is asymmetric, i.e. it hands the anticipating class information
     the myopic class could have used. The planted term is a function of
     (step, bin, action) added to the one shared cost array, so symmetry is
     structural; `test_plant_is_symmetric_in_cost` pins that it is applied to
     the same array both solvers read, and to every cell alike.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, filename):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = load("tsar_cal", "analyze_tsar_cefr_headroom_calibration.py")
mref = sys.modules["analyze_tsar_cefr_moving_reference"]
d25 = sys.modules["analyze_tsar_cefr_d25"]


def make_cell(text_id, source_id, target, seed, lo=3.6, hi=5.0):
    rng = np.random.default_rng(seed)
    cell = d25.Cell(text_id, source_id, target)
    for idx in range(d25.N_NODES):
        cell.ell[idx] = float(rng.uniform(lo, hi))
    cell.termsq[d25.OFFSETS[d25.DEPTH] :] = 0.0
    return cell


def fixture_cells(n=4, seed0=100):
    return [make_cell(f"c{i}", f"s{i}", "B1", seed0 + i) for i in range(n)]


def test_pistar_is_not_step_invariant():
    """No single bin->action table can match pistar at every step.

    If one could, the plant would cost both classes the same and measure nothing.
    """
    for b in range(3):
        actions = {cal.pistar(t, b) for t in range(1, d25.DEPTH + 1)}
        assert len(actions) > 1, f"bin {b}: pistar is constant across steps"


def test_plant_none_is_byte_identical():
    """The published probe path must not move now that `plant` exists."""
    cells_by_target = {"B1": fixture_cells()}
    base = mref.run_schedule(cells_by_target, "constant", 3, verify=False)
    with_arg = mref.run_schedule(cells_by_target, "constant", 3, verify=False, plant=None)
    assert base == with_arg


def test_plant_is_symmetric_in_cost():
    """The planted term touches the one shared cost array, every cell alike."""
    cells = fixture_cells()
    g0 = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["constant"])
    g1 = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["constant"])
    delta = 0.05
    mref.plant_step_dependent_cost(g1, delta, cal.pistar)
    diff = g1.cost - g0.cost
    off = {round(float(v), 12) for v in np.unique(diff)}
    assert off <= {0.0, round(delta, 12)}, f"planting wrote values other than 0/delta: {off}"
    # bins are what the policies see; planting must not have moved them
    assert np.array_equal(g0.binv, g1.binv)
    # every cell pays on the same set of nodes: the term depends on (t, bin, action) only
    for i in range(1, g1.n):
        rows_equal = np.array_equal((diff[i] > 0), (diff[0] > 0))
        same_bins = np.array_equal(g1.binv[i], g1.binv[0])
        assert rows_equal or not same_bins


def test_true_headroom_grows_with_delta():
    """A large plant must buy headroom, but the curve is NOT monotone in delta.

    Both J_myopic and J_anticipating are a min over tables, so each is concave in
    delta, and a difference of two concave functions can fall before it rises.
    This fixture does fall between delta=0 and delta=0.05. That is a property of
    the knob, not a defect, and it is why the report plots the LOO estimate
    against the EXACTLY SOLVED true headroom rather than against delta.
    """
    cells_by_target = {"B1": fixture_cells()}
    got = [cal.true_headroom(cells_by_target, d, 3)["pooled_G_plan"]
           for d in (0.0, 0.05, 0.20, 1.0)]
    assert got[-1] > got[0] + 1e-6, f"a large plant bought no headroom: {got}"
    assert min(got) >= -1e-12, f"true headroom went negative: {got}"


def test_true_headroom_is_the_population_optimum_not_a_fit():
    """Truth is solved on all cells by the same exact solvers, so it cannot be beaten."""
    cells = fixture_cells()
    g = mref.MovingRefGroup(cells, 3, mref.SCHEDULES["constant"])
    mref.plant_step_dependent_cost(g, 0.05, cal.pistar)
    members = np.arange(g.n)
    pos3, dec = mref.enumerate_prefixes_running(g)
    _, j_an = mref.solve_anticipating(g, pos3, dec, members)
    for combo in itertools.product(range(d25.N_ACTIONS), repeat=3):
        table = np.tile(np.array(combo, dtype=np.int8), (d25.DEPTH, 1))
        j = float(np.mean([g.walk_cost(table, int(i), False) for i in members]))
        assert j_an <= j + 1e-12


def test_refuses_to_overwrite_output(tmp_path):
    out = tmp_path / "already.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        cal.main(["--tree", str(tmp_path / "nope.jsonl"), "--out", str(out)])
