import importlib.util
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_b_identifiability_bootstrap", REPO / "scripts" / "b_identifiability_bootstrap.py")
bib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bib)


def _units(gain: float, n_traj: int = 40, n_items: int = 10, T: int = 12, seed: int = 0):
    """y_(t+1) = 0.6 y_t + gain * v + noise, v randomised per turn."""
    rng = np.random.default_rng(seed)
    units = []
    for i in range(n_traj):
        y = [rng.normal()]
        v = rng.integers(0, 2, size=T).astype(float)
        for t in range(T - 1):
            y.append(0.6 * y[-1] + gain * v[t] + 0.1 * rng.normal())
        Z = np.asarray(y[:-1]).reshape(-1, 1)
        Z_next = np.asarray(y[1:]).reshape(-1, 1)
        units.append((f"t{i}", f"item{i % n_items}",
                      {"Z": Z, "V": v[:-1].reshape(-1, 1), "Z_next": Z_next, "Y": Z[:, 0]}))
    return units


def _main_elements(units):
    B, CB = bib._fit(bib._stack([u[2] for u in units]))
    _, CB_draws, _ = bib._bootstrap(units, lambda u: u[0], seed=0, n_resamples=200)
    return CB, bib._intervals(CB_draws, CB)


def test_a_real_command_effect_is_pinned():
    CB, main = _main_elements(_units(gain=0.5))
    assert abs(CB[0] - 0.5) < 0.05
    assert main[0]["excludes_zero"] and main[0]["sign_stability"] == 1.0
    assert bib._verdict(True, main)


def test_a_null_command_effect_is_rarely_usable():
    """No effect in the data -> a 95% interval excludes 0 about 5% of the time.
    Checked as a rate over seeds, not on one seed: any single seed is a coin
    with a 5% face, and picking one that passes would test nothing."""
    passed = [bib._verdict(True, _main_elements(_units(gain=0.0, seed=s))[1]) for s in range(10)]
    assert sum(passed) <= 2


def test_an_unidentified_rank_check_vetoes_a_pinned_interval():
    _, main = _main_elements(_units(gain=0.5))
    assert not bib._verdict(False, main)


def test_sign_stability_counts_agreement_with_the_point_estimate():
    draws = np.array([[1.0], [2.0], [-1.0], [3.0]])
    (only,) = bib._intervals(draws, np.array([0.5]))
    assert only["sign_stability"] == 0.75
