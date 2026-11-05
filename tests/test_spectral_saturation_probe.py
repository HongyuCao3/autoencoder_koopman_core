import importlib.util
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_spectral_saturation_probe", ROOT / "scripts" / "spectral_saturation_probe.py")
ssp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ssp)

COLUMNS = ("normalized_output",), ("effective_norm",)


def _frame(step, n_traj=60, n_turns=10, seed=0):
    """One scalar trajectory per row block; `step(y_hist, r, level, rng)` gives y_(t+1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_traj):
        r = float(rng.uniform(0.2, 0.9))
        level = float(rng.normal())
        hist = [level + 0.1 * float(rng.standard_normal()), float(rng.uniform(0.0, 1.0))]
        split = ("train", "validation", "test")[i % 3]
        for turn in range(1, n_turns + 1):
            rows.append({"trajectory_id": f"traj_{i:02d}", "topic": f"t{i}", "topic_split": split,
                         "turn": turn, "normalized_output": hist[0], "effective_norm": r})
            hist = [step(hist, r, level, rng), hist[0]]
    return pd.DataFrame(rows)


def test_ar2_third_eigenvalue_is_dead():
    """Spec 4.3-1: a true AR(2) has two modes; the third one a depth-2 state
    adds must come out near zero.

    Only the point estimate is asserted. The bootstrap upper bound that (a)
    reads does NOT clear tau here even with 100 training trajectories (0.19):
    the modulus of a mode at zero is biased upward under resampling, so (a)
    has little power at this project's sample sizes. Recorded in the results
    doc rather than tuned away -- tau and the statistic are pre-signed."""
    frame = _frame(lambda h, r, lv, rng: 1.2 * h[0] - 0.45 * h[1] + 0.25 * r + 0.1 * rng.standard_normal())
    out = ssp.run_core_frame(frame, *COLUMNS, "ar2", n_resamples=50)
    L2 = out["depths"]["L_2"]
    assert L2["modulus"][2] < 0.05
    assert abs(L2["modulus"][0] - np.sqrt(0.45)) < 0.05      # 0.6 +/- 0.3i


def test_item_level_plus_noise_shows_a_unit_mode_and_one_mode_suffices():
    """Spec 4.3-2: persistent level + white noise -> a mode at ~1 (the level),
    and cutting K to that one mode does not lose multi-step skill."""
    frame = _frame(lambda h, r, lv, rng: lv + 0.1 * rng.standard_normal())
    out = ssp.run_core_frame(frame, *COLUMNS, "level", n_resamples=50)
    L3 = out["depths"]["L_3"]
    assert L3["modulus"][0] > 0.95
    skill = {k: v["skill_h"] for k, v in L3["truncated_skill"].items()}
    assert skill["k_1"] >= skill["k_4"] - 0.01


def test_new_modes_are_the_unmatched_ones():
    prev = np.array([0.9, 0.5])
    cur = np.array([0.91, 0.02, 0.49])
    fresh, dist = ssp.new_modes(prev, cur)
    assert np.allclose(fresh, [0.02]) and np.allclose(dist, [0.48])


def test_full_rank_truncation_is_the_identity():
    K = np.array([[0.5, 0.2, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(ssp.truncate(K, 3), K)
