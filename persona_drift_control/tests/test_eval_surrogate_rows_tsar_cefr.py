"""Table 1's eleventh column. The load-bearing test is the planted pair at the
bottom: this column exists to answer whether row 3 (`ours - withheld u`) is
non-zero, and a script that cannot report non-zero when the action demonstrably
drives the state would make a zero unreadable -- the `run_config_guard`
principle (a guard that passes silently hides exactly what it was built for).
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "eval_surrogate_rows_tsar_cefr", SCRIPTS / "eval_surrogate_rows_tsar_cefr.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def make_rows(n_sources=12, seeds=(0, 1), targets=("A2", "B1"), steps=6,
              gain=None, rng_seed=0, actions=None):
    """Synthetic rows in the `tsar_cefr` schema.

    `gain` maps an action to its one-step effect on `ell`. `None` means the
    action does nothing -- the negative control.
    """
    rng = np.random.default_rng(rng_seed)
    acts = actions or ["step_down", "half_step_down", "paraphrase", "copy"]
    rows = []
    for si in range(n_sources):
        src_level = 3.0 + 0.5 * rng.standard_normal()
        for target in targets:
            for seed in seeds:
                ell = src_level
                for t in range(1, steps + 1):
                    a = acts[rng.integers(len(acts))]
                    ell = 0.9 * ell + (gain.get(a, 0.0) if gain else 0.0) + 0.02 * rng.standard_normal()
                    rows.append({
                        "text_id": f"{si}-{target.lower()}", "source_id": str(si),
                        "target_cefr": target, "seed": seed, "step": t, "action": a,
                        "level_expected": ell, "meaning_to_source": 0.9 - 0.01 * t,
                        "source_level_expected": src_level,
                    })
    return rows


def test_exclusion_and_step_zero_conventions():
    rows = make_rows(n_sources=3)
    rows.append(dict(rows[0], text_id=mod.EXCLUDED_TEXT_IDS[0]))
    trajs = mod.build_trajectories(rows)
    assert all(t["text_id"] != mod.EXCLUDED_TEXT_IDS[0] for t in trajs.values())
    one = next(iter(trajs.values()))
    assert one["ell"][0] == pytest.approx(float(one["ell"][0]))
    assert one["s"][0] == 1.0, "s_0 is the source's similarity to itself"
    assert 0 not in one["u"], "step 0 has no action that produced it"


def test_action_is_paired_with_the_state_it_produced():
    rows = make_rows(n_sources=3)
    traj = next(iter(mod.build_trajectories(rows).values()))
    trs = mod.transitions(traj, mod.LAG)
    for tr in trs:
        np.testing.assert_array_equal(tr["v"], traj["u"][tr["turn_next"]])
        np.testing.assert_allclose(tr["z_next"][mod.ELL_INDEX], traj["ell"][tr["turn_next"]])
    assert len(trs) == traj["max_t"] - mod.LAG


def test_action_is_never_collapsed_to_a_scalar():
    assert len(mod.U_COLS) == 3 and mod.REFERENCE_ACTION not in mod.ACTIONS
    assert "turn_next" in mod.EXOGENOUS, "trivial_nulls raises without it"


def test_every_row_is_scored_on_identical_windows():
    rows = make_rows(n_sources=3)
    traj = next(iter(mod.build_trajectories(rows).values()))

    class Fake:
        def __init__(self, lag): self.lag = lag
        def step(self, z, v): return np.asarray(z, float) * 0.5
        def readout(self, z): return float(np.asarray(z, float)[0])

    deep = mod.rollout(Fake(mod.LAG), traj, mod.LAG, False)
    flat = mod.rollout(Fake(0), traj, 0, False)
    assert [(r["turn"], r["step"]) for r in deep] == [(r["turn"], r["step"]) for r in flat]
    assert len(deep) == mod.HORIZON * (traj["max_t"] - mod.HORIZON - mod.LAG + 1)


def test_no_control_row_really_withholds_the_action():
    class Inner:
        def step(self, z, v): return np.asarray(z, float) + np.sum(np.asarray(v, float))
        def readout(self, z): return float(np.asarray(z, float)[0])

    blind = mod._NoControl(Inner())
    z = np.array([1.0, 0.0])
    a = blind.step(z, np.array([1.0, 0.0, 0.0]))
    b = blind.step(z, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, z)


@pytest.mark.parametrize("planted,expect_positive", [
    ({"step_down": -0.50, "half_step_down": -0.25, "paraphrase": -0.05}, True),
    (None, False),
])
def test_planted_controls_row_three(monkeypatch, planted, expect_positive):
    """Positive control: when the action drives `ell`, withholding it must cost
    measurable skill. Negative control: when it does not, row 3 must sit on 0."""
    monkeypatch.setattr(mod, "AE_EPOCHS", 40)
    monkeypatch.setattr(mod, "LSTM_HIDDEN_CANDIDATES", (4,))
    rows = make_rows(n_sources=15, gain=planted, rng_seed=7)
    trajs = [v for _, v in sorted(mod.build_trajectories(rows).items())]
    result = mod.run_on_trajectories(trajs, bootstrap_seed=0)
    c = result["contrasts"]["delay_linear_control_minus_delay_linear_no_control"]
    if expect_positive:
        assert c["point"] > 0 and c["excludes_zero"], f"planted action not detected: {c}"
    else:
        assert not c["excludes_zero"], f"row 3 fired on a null action: {c}"


def test_target_rank_is_per_trajectory_not_per_source():
    """One `source_id` carries both target levels; the null's exogenous block
    must not collapse them (the A2 and B1 runs of a source are different
    environments, and `item` is the bootstrap unit, not the environment)."""
    rows = make_rows(n_sources=4, targets=("A2", "B1"), seeds=(0,))
    trajs = mod.build_trajectories(rows)
    by_source = {}
    for t in trajs.values():
        by_source.setdefault(t["source_id"], set()).add(t["target_level_rank"])
    assert all(len(v) == 2 for v in by_source.values()), "fixture must exercise both targets"
    assert mod.CEFR_RANK["A2"] != mod.CEFR_RANK["B1"]
