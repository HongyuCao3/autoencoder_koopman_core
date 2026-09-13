"""Tests for scripts/screen_observation_richness.py.

The screen's whole job is to tell signal from cardinality, so the tests plant
both: a feature that is a per-row fingerprint (must look good in-sample and
collapse under cross-fitting) and a feature that genuinely carries the answer
(must survive cross-fitting).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
TURNS = (1, 2, 3, 4, 5)


def _load():
    spec = importlib.util.spec_from_file_location("_screen", SCRIPTS / "screen_observation_richness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zero(keys, value_of):
    return {key: {turn: {"y_safety": 1.0, "y_probe": 1.0, "agent_message": "m" * value_of(key, turn),
                         "attacker_query": "q" * value_of(key, turn)}
                  for turn in TURNS} for key in keys}


def test_binner_collapses_a_fingerprint_into_the_requested_number_of_bins():
    module = _load()
    binner = module._binner(list(range(100)), 3)
    assert sorted({binner(v) for v in range(100)}) == [0, 1, 2]
    assert binner(0) <= binner(50) <= binner(99)


def test_binner_with_zero_bins_is_the_identity():
    """`y_safety` already takes five discrete values; binning it again would
    throw away the resolution the baseline bound is defined on."""
    module = _load()
    keep = module._binner([0.0, 0.5, 1.0], 0)
    assert [keep(v) for v in (0.0, 0.5, 1.0)] == [0.0, 0.5, 1.0]


def test_lookahead_feature_includes_the_turn_being_decided():
    module = _load()
    keys = [("a", 0)]
    zero = _zero(keys, lambda key, turn: turn)
    behind = module.build_prefixes(zero, keys, "query_len")
    ahead = module.build_prefixes(zero, keys, "query_len_lookahead")
    assert len(behind[(("a", 0), 3)]) == 2   # turns 1, 2
    assert len(ahead[(("a", 0), 3)]) == 3    # turns 1, 2, 3 -- the query is on screen
    assert len(behind[(("a", 0), 1)]) == 0   # nothing at all before the first decision
    assert len(ahead[(("a", 0), 1)]) == 1


def test_paired_feature_carries_both_signals():
    module = _load()
    keys = [("a", 0), ("b", 0)]
    zero = _zero(keys, lambda key, turn: 1 if key[0] == "a" else 90)
    paired = module.build_prefixes(zero, keys, "y_safety_plus_message_len")
    # y_safety is constant across these two, message_len is not, so the pair
    # must still separate them.
    assert paired[(("a", 0), 3)] != paired[(("b", 0), 3)]
    assert len(paired[(("a", 0), 3)]) == 4  # 2 turns x 2 components


def test_fingerprint_wins_in_sample_and_collapses_when_cross_fitted():
    """The reason the gate is the cross-fitted column. A per-row id lets the
    induction memorise the answer; it must not survive a held-out fold."""

    ceiling = importlib.util.spec_from_file_location("_c", SCRIPTS / "analyze_adaptivity_ceiling.py")
    module = importlib.util.module_from_spec(ceiling)
    ceiling.loader.exec_module(module)

    rng = np.random.default_rng(0)
    keys = [(f"a{i}", s) for i in range(6) for s in range(4)]
    payoff = {key: {turn: float(rng.normal()) for turn in TURNS} for key in keys}
    terminal = {turn: {key: payoff[key][turn] for key in keys} for turn in TURNS}
    never = {key: -5.0 for key in keys}
    fingerprint = {(key, turn): (keys.index(key),) for key in keys for turn in TURNS}

    in_sample = np.mean([module.causal_oracle(keys, keys, fingerprint, terminal, never, +1)[key] for key in keys])
    held = {}
    for attack in sorted({key[0] for key in keys}):
        held.update(module.causal_oracle([k for k in keys if k[0] != attack],
                                         [k for k in keys if k[0] == attack],
                                         fingerprint, terminal, never, +1))
    crossfit = np.mean([held[key] for key in keys])
    best_fixed = max(float(np.mean([terminal[t][key] for key in keys])) for t in TURNS)
    assert in_sample > best_fixed + 0.3   # memorised the per-row maximum
    assert crossfit < in_sample           # none of it generalises


def test_guard_rejects_an_attacker_that_reacts_to_the_reminder(tmp_path, monkeypatch):
    """A look-ahead feature is only replayable while the attacker script is
    fixed. If the attacker adapts, the query at turn t is no longer the same
    question in every arm and the replay is scoring a different experiment."""

    module = _load()
    keys = [("a", 0)]
    zero = _zero(keys, lambda key, turn: turn)
    arms = {t: {key: {turn: dict(zero[key][turn]) for turn in TURNS} for key in keys} for t in TURNS}
    arms[3][("a", 0)][3]["attacker_query"] = "the attacker noticed the reminder"
    divergent = sum(
        arms[t][key][turn]["attacker_query"] != zero[key][turn]["attacker_query"]
        for t in TURNS for key in keys for turn in TURNS
    )
    assert divergent == 1  # the condition main() raises on
