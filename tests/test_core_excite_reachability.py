import importlib.util
import pathlib

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "core_excite_reach", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "core_excite_reachability.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_steady_state_range_matches_fixed_point():
    K = np.zeros((4, 4)); K[0] = [0.1, 0.05, 0.0, 0.05]
    B = np.array([[0.6], [0], [0], [0]]); b = np.array([0.02, 0, 0, 0])
    lo, hi = mod.steady_state_range(K, B, b)
    y_hi = (0.6 + 0.02) / (1 - 0.2)
    assert np.isclose(hi, 5 + 95 * y_hi) and np.isclose(lo, 5 + 95 * 0.02 / 0.8)


def test_seed_report_recovers_ceiling_on_synthetic_static_gain():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        split = ("train", "val", "test")[i % 3]
        for t in range(1, 11):
            c = rng.uniform(0, 1)
            y = 0.7 * c + 0.01 * rng.standard_normal()
            rows.append({"trajectory_id": f"t{i}", "turn": t, "topic_split": split, "command_norm": c,
                         "normalized_output": y, "command_raw_count": 5 + 95 * c, "measured_raw_count": 5 + 95 * y})
    rep = mod.seed_report(rows)
    assert abs(rep["steady_state_words"]["c=1 (100 words)"] - (5 + 95 * 0.7)) < 1.5
    for k, v in rep["heldout_one_step"]["predicted"].items():
        assert abs(v["mean"] - rep["heldout_one_step"]["observed"][k]["mean"]) < 1.0
