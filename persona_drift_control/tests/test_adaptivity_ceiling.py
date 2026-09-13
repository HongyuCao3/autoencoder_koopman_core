"""Tests for scripts/analyze_adaptivity_ceiling.py.

The bounds in that script are only meaningful if two things hold: the oracle
really is an oracle (it can never do worse than a policy it is allowed to
imitate), and it really is CAUSAL (it cannot act on a distinction it cannot
observe). Both get planted controls here -- the first invariant is what caught
a sign bug that built the worst policy on the one objective where lower is
better, while still printing it under the name `oracle`.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_adaptivity_ceiling.py"
TURNS = (1, 2, 3, 4, 5)


def _load():
    spec = importlib.util.spec_from_file_location("_ceiling", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _problem(keys, prefix_of, terminal_of, never_of):
    """Builds the three plain dicts `causal_oracle` consumes."""
    prefix = {(key, turn): prefix_of(key)[: turn - 1] for key in keys for turn in TURNS}
    terminal = {turn: {key: terminal_of(key)[turn] for key in keys} for turn in TURNS}
    never = {key: never_of(key) for key in keys}
    return prefix, terminal, never


def _best_fixed_level(terminal, keys, sign):
    levels = {turn: float(np.mean([terminal[turn][key] for key in keys])) for turn in TURNS}
    return max(levels.values()) if sign > 0 else min(levels.values())


@pytest.mark.parametrize("sign", [+1, -1])
def test_in_sample_oracle_is_never_worse_than_the_best_fixed_schedule(sign):
    """The invariant that makes it an oracle: firing at turn k for everyone is
    inside its action set, so it can always imitate the best fixed schedule.
    A maximising induction run on a LOSS objective violates this -- which is
    exactly how the sign bug showed itself."""

    module = _load()
    rng = np.random.default_rng(0)
    keys = [(f"a{i}", s) for i in range(6) for s in range(5)]
    payoff = {key: {turn: float(rng.normal()) for turn in TURNS} for key in keys}
    prefix, terminal, never = _problem(
        keys,
        lambda key: (1.0, float(hash(key) % 3), 0.5, 0.5),
        lambda key: payoff[key],
        lambda key: float(rng.normal()),
    )
    replayed = module.causal_oracle(keys, keys, prefix, terminal, never, sign)
    level = float(np.mean([replayed[key] for key in keys]))
    best_fixed = _best_fixed_level(terminal, keys, sign)
    assert sign * level >= sign * best_fixed - 1e-12


def test_oracle_cannot_act_on_a_distinction_it_cannot_observe():
    """Negative control. Two halves want opposite turns, but their observable
    prefixes are identical, so no causal policy can tell them apart and the
    bound must collapse to the best fixed schedule."""

    module = _load()
    keys = [("a", s) for s in range(10)] + [("b", s) for s in range(10)]
    wants_early = set(keys[:10])

    def payoff(key):
        return {1: 0.0, 2: 1.0 if key in wants_early else 0.0, 3: 0.0, 4: 0.0,
                5: 0.0 if key in wants_early else 1.0}

    prefix, terminal, never = _problem(keys, lambda key: (1.0, 1.0, 1.0, 1.0), payoff, lambda key: 0.0)
    replayed = module.causal_oracle(keys, keys, prefix, terminal, never, +1)
    level = float(np.mean([replayed[key] for key in keys]))
    assert level == pytest.approx(_best_fixed_level(terminal, keys, +1))
    assert level == pytest.approx(0.5)  # either fixed turn serves exactly half


def test_oracle_collects_the_gain_when_the_prefix_does_reveal_it():
    """Positive control, same payoffs as above but now the two halves differ
    in y_1, so the decision at turn 2 can condition on it."""

    module = _load()
    keys = [("a", s) for s in range(10)] + [("b", s) for s in range(10)]
    wants_early = set(keys[:10])

    def payoff(key):
        return {1: 0.0, 2: 1.0 if key in wants_early else 0.0, 3: 0.0, 4: 0.0,
                5: 0.0 if key in wants_early else 1.0}

    prefix, terminal, never = _problem(
        keys, lambda key: (1.0 if key in wants_early else 0.25, 1.0, 1.0, 1.0), payoff, lambda key: 0.0)
    replayed = module.causal_oracle(keys, keys, prefix, terminal, never, +1)
    assert float(np.mean([replayed[key] for key in keys])) == pytest.approx(1.0)


def test_unseen_prefix_backs_off_instead_of_crashing():
    """Cross-fitting hands the replay histories the fit never saw; a policy at
    deployment has to answer anyway."""

    module = _load()
    fit = [("a", s) for s in range(5)]
    held = [("b", 0)]
    keys = fit + held
    prefix, terminal, never = _problem(
        keys,
        lambda key: (1.0, 1.0, 1.0, 1.0) if key[0] == "a" else (0.0, 0.0, 0.0, 0.0),
        lambda key: {turn: 1.0 if turn == 3 else 0.0 for turn in TURNS},
        lambda key: 0.0,
    )
    replayed = module.causal_oracle(fit, held, prefix, terminal, never, +1)
    assert replayed[("b", 0)] == 1.0  # backed off to the fitted turn-3 decision


# ------------------------------------------------------------------ the guard

def _rows(directory: pathlib.Path, fire_turn: int | None, message=lambda key, turn: f"m{key}{turn}"):
    rows = []
    for attack in ("a", "b"):
        for seed in range(2):
            for turn in TURNS:
                rows.append({"attack_id": attack, "seed": seed, "turn": turn, "y_safety": 1.0,
                             "u_remind": int(turn == fire_turn), "agent_message": message((attack, seed), turn)})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trajectories.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return {(row["attack_id"], row["seed"]): {} for row in rows}


def _index(directory: pathlib.Path):
    out: dict = {}
    for line in (directory / "trajectories.jsonl").read_text().splitlines():
        row = json.loads(line)
        out.setdefault((row["attack_id"], row["seed"]), {})[row["turn"]] = row
    return out


def test_guard_accepts_arms_whose_prefixes_match(tmp_path):
    module = _load()
    _rows(tmp_path / "zero", None)
    for turn in TURNS:
        _rows(tmp_path / f"f{turn}", turn)
    result = module.guard_prefix_identity(
        _index(tmp_path / "zero"), {turn: _index(tmp_path / f"f{turn}") for turn in TURNS})
    assert result["all_identical"]
    assert result["pre_reminder_turns_checked"] == 4 * (0 + 1 + 2 + 3 + 4)


def test_guard_rejects_a_prefix_that_diverges_before_the_reminder(tmp_path):
    """If an arm's pre-reminder turns differ from zero_control, the observed
    history at a decision point is not the zero-control prefix and every bound
    silently becomes a different quantity."""

    module = _load()
    _rows(tmp_path / "zero", None)
    for turn in TURNS:
        _rows(tmp_path / f"f{turn}", turn)
    _rows(tmp_path / "f5", 5, message=lambda key, turn: "diverged" if turn == 2 else f"m{key}{turn}")
    with pytest.raises(SystemExit, match="pre-reminder turns differ"):
        module.guard_prefix_identity(
            _index(tmp_path / "zero"), {turn: _index(tmp_path / f"f{turn}") for turn in TURNS})


def test_guard_rejects_an_arm_that_fires_on_the_wrong_turn(tmp_path):
    module = _load()
    _rows(tmp_path / "zero", None)
    for turn in TURNS:
        _rows(tmp_path / f"f{turn}", turn)
    _rows(tmp_path / "f3", 4)  # labelled t3, actually fires at t4
    with pytest.raises(SystemExit, match="mislabelled"):
        module.guard_prefix_identity(
            _index(tmp_path / "zero"), {turn: _index(tmp_path / f"f{turn}") for turn in TURNS})
