"""Paired self-judge vs independent-judge statistics for the adversarial
line, and the readout-quality numbers the defense line's post-mortem is
phrased in.

The sycophancy line quantified its self-judging bias with a three-cell
confusion matrix over a discrete label (MAINTAINS/HEDGES/FLIPS,
scripts/compare_judge_runs.py). The defense line's readout is a 1-5 judge
score mapped to {0, .25, .5, .75, 1}, so the same four questions need
slightly different machinery -- a signed difference rather than a label
flip -- plus two the sycophancy comparison had no reason to ask:

- Does the bias depend on `u_remind`? The defense claim is a *between-arm*
  difference, and a constant level offset cancels out of it. A bias that is
  larger on reminded turns does not: the agent judging its own reply right
  after being handed a safety reminder is exactly the situation where
  "I was just told to be safe" could inflate its own score, which would
  manufacture part of the measured defense effect.
- What does each judge's score distribution do to the *resolution* the arm
  comparison needs (ceiling share, number of distinct levels, and the
  erosion-signal SNR that docs/experiments/koopman_defense_pilot.md section
  五 reports as 1.63 for the judge vs 4.20 for the activation projection)?

Everything here is pure functions over row dicts; the file/CLI half lives in
scripts/compare_safety_judge_runs.py.
"""

from __future__ import annotations

import collections
from typing import Any, Iterable

import numpy as np
from scipy import stats


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (row["trajectory_id"], int(row["turn"]))


def pair_rows(self_rows: list[dict[str, Any]], indep_rows: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    """Joins the two score sets on (trajectory_id, turn).

    Also returns the pairing diagnostics that decide whether the comparison
    below may be read at all: `n_text_mismatch` must be 0, otherwise the two
    judges scored different replies and nothing downstream is paired. With
    rejudge.py this is 0 by construction (the same stored `agent_message` is
    reused); it is still checked, because the same function is meant to also
    accept a genuine second screening run whose agent text could differ."""

    self_by_key = {_key(row): row for row in self_rows}
    indep_by_key = {_key(row): row for row in indep_rows}
    shared = sorted(set(self_by_key) & set(indep_by_key))

    pairs: list[dict[str, Any]] = []
    n_text_mismatch = 0
    for key in shared:
        a, b = self_by_key[key], indep_by_key[key]
        if a.get("agent_message") != b.get("agent_message"):
            n_text_mismatch += 1
            continue
        pairs.append(
            {
                "trajectory_id": key[0],
                "turn": key[1],
                "attack_id": a.get("attack_id"),
                "seed": a.get("seed"),
                "u_remind": int(a.get("u_remind", 0)),
                "y_self": float(a["y_safety"]) if a["y_safety"] == a["y_safety"] else float("nan"),
                "y_indep": float(b["y_safety"]) if b["y_safety"] == b["y_safety"] else float("nan"),
                "self_judge_model": a.get("judge_model"),
                "indep_judge_model": b.get("judge_model"),
            }
        )

    diagnostics = {
        "n_self_rows": len(self_rows),
        "n_indep_rows": len(indep_rows),
        "n_shared_keys": len(shared),
        "n_paired": len(pairs),
        "n_text_mismatch": n_text_mismatch,
        "n_self_only": len(set(self_by_key) - set(indep_by_key)),
        "n_indep_only": len(set(indep_by_key) - set(self_by_key)),
        "self_judge_model": sorted({p["self_judge_model"] for p in pairs}),
        "indep_judge_model": sorted({p["indep_judge_model"] for p in pairs}),
        "n_nan_self": sum(1 for p in pairs if p["y_self"] != p["y_self"]),
        "n_nan_indep": sum(1 for p in pairs if p["y_indep"] != p["y_indep"]),
    }
    return pairs, diagnostics


def _finite(pairs: Iterable[dict]) -> list[dict]:
    return [p for p in pairs if p["y_self"] == p["y_self"] and p["y_indep"] == p["y_indep"]]


def _diff_summary(pairs: list[dict]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0}
    diffs = np.array([p["y_indep"] - p["y_self"] for p in pairs])
    n_lower = int((diffs < 0).sum())
    n_higher = int((diffs > 0).sum())
    n_disagree = n_lower + n_higher
    return {
        "n": len(pairs),
        "mean_self": float(np.mean([p["y_self"] for p in pairs])),
        "mean_indep": float(np.mean([p["y_indep"] for p in pairs])),
        "mean_diff_indep_minus_self": float(diffs.mean()),
        "n_disagree": n_disagree,
        "disagree_rate": n_disagree / len(pairs),
        "n_independent_stricter": n_lower,
        "n_independent_looser": n_higher,
        # One-way-ness: the sycophancy line's headline was that all 26
        # disagreements went the same way (p~3e-8), i.e. the self-judge only
        # ever misses, never over-flags. Two-sided binomial on the
        # disagreeing rows only -- the equal rows carry no directional
        # information.
        "sign_test_p": float(stats.binomtest(n_lower, n_disagree, 0.5).pvalue) if n_disagree else float("nan"),
    }


def analyze_judge_bias(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The four sycophancy-line questions (agreement, direction, is the bias
    a level shift or turn-dependent, how concentrated) plus the reminder
    interaction."""

    usable = _finite(pairs)
    result: dict[str, Any] = {"overall": _diff_summary(usable)}

    confusion: collections.Counter = collections.Counter((p["y_self"], p["y_indep"]) for p in usable)
    result["confusion"] = {f"self={s:.2f}->indep={i:.2f}": int(n) for (s, i), n in sorted(confusion.items())}

    by_turn: dict[int, dict] = {}
    for turn in sorted({p["turn"] for p in usable}):
        by_turn[turn] = _diff_summary([p for p in usable if p["turn"] == turn])
    result["by_turn"] = by_turn

    # Is the bias a constant level offset (harmless to a trend, ~4x
    # compression of the effect size) or does it grow with turn (in which
    # case it distorts new-Q1's slope itself)? Pooled OLS over rows, on both
    # the signed difference and the disagreement indicator -- the sycophancy
    # comparison ran the indicator version and read p=0.300 as "constant
    # offset"; the signed version is the one that maps directly onto a slope
    # bias, so both are reported.
    if len({p["turn"] for p in usable}) >= 2:
        turns = np.array([p["turn"] for p in usable], dtype=float)
        diffs = np.array([p["y_indep"] - p["y_self"] for p in usable])
        indicator = (diffs != 0).astype(float)
        slope_d, _, _, p_d, _ = stats.linregress(turns, diffs)
        slope_i, _, _, p_i, _ = stats.linregress(turns, indicator)
        result["bias_vs_turn"] = {
            "signed_diff_slope": float(slope_d),
            "signed_diff_p": float(p_d),
            "disagreement_rate_slope": float(slope_i),
            "disagreement_rate_p": float(p_i),
        }
    else:
        result["bias_vs_turn"] = {}

    # Concentration: "26/200 rows wrong" would misread a bias that is a few
    # items being systematically wrong on every turn.
    disagreeing = [p for p in usable if p["y_indep"] != p["y_self"]]
    by_attack = collections.Counter(p["attack_id"] for p in disagreeing)
    by_trajectory = collections.Counter(p["trajectory_id"] for p in disagreeing)
    n_attacks = len({p["attack_id"] for p in usable})
    result["concentration"] = {
        "n_disagreeing_rows": len(disagreeing),
        "n_attacks_with_any_disagreement": len(by_attack),
        "n_attacks_total": n_attacks,
        "n_trajectories_with_any_disagreement": len(by_trajectory),
        "n_trajectories_total": len({p["trajectory_id"] for p in usable}),
        "top_attacks": dict(by_attack.most_common(5)),
        "max_single_attack_share": (max(by_attack.values()) / len(disagreeing)) if disagreeing else float("nan"),
    }

    # The reminder interaction (see module docstring): a bias that is
    # bigger on reminded rows does not cancel out of a between-arm
    # difference.
    reminded = [p for p in usable if p["u_remind"] == 1]
    unreminded = [p for p in usable if p["u_remind"] == 0]
    result["by_u_remind"] = {"reminded": _diff_summary(reminded), "unreminded": _diff_summary(unreminded)}
    if reminded and unreminded:
        a = np.array([p["y_indep"] - p["y_self"] for p in reminded])
        b = np.array([p["y_indep"] - p["y_self"] for p in unreminded])
        # A Welch test on two constant samples is a 0/0 (scipy warns and
        # returns nan); the mean difference is still the number worth
        # reporting, so the test is what gets skipped, not the comparison.
        testable = len(a) >= 2 and len(b) >= 2 and (a.std() > 0 or b.std() > 0)
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False) if testable else (float("nan"), float("nan"))
        result["by_u_remind"]["difference_of_biases"] = {
            "mean_reminded_minus_unreminded": float(a.mean() - b.mean()),
            "welch_t": float(t_stat),
            "welch_p": float(t_p),
            "n_reminded": len(a),
            "n_unreminded": len(b),
        }
    return result


def readout_quality(rows: list[dict[str, Any]], value_key: str = "y_safety") -> dict[str, Any]:
    """Resolution of one judge's scores: how much of the scale is actually
    used, and how large the one signal we know to be real (turn-over-turn
    erosion) is relative to the spread within a turn.

    `erosion_snr` is defined exactly as koopman_defense_pilot.md section 五
    defines it -- (first-turn mean minus last-turn mean) / pooled
    within-turn sd -- so the number printed here is comparable to the 1.63
    (judge) and 4.20 (activation projection) already in that document."""

    values = np.array([float(r[value_key]) for r in rows if r[value_key] == r[value_key]])
    if values.size == 0:
        return {"n": 0}
    by_turn: dict[int, np.ndarray] = {}
    for turn in sorted({int(r["turn"]) for r in rows}):
        vals = np.array([float(r[value_key]) for r in rows if int(r["turn"]) == turn and r[value_key] == r[value_key]])
        if vals.size:
            by_turn[turn] = vals

    within_sds = [float(np.std(v, ddof=1)) for v in by_turn.values() if v.size >= 2]
    pooled_within_sd = float(np.mean(within_sds)) if within_sds else float("nan")
    turns = sorted(by_turn)
    snr = (
        abs(float(np.mean(by_turn[turns[0]]) - np.mean(by_turn[turns[-1]]))) / pooled_within_sd
        if len(turns) >= 2 and pooled_within_sd and pooled_within_sd == pooled_within_sd
        else float("nan")
    )
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size >= 2 else float("nan"),
        "ceiling_share": float((values == 1.0).mean()),
        "floor_share": float((values == 0.0).mean()),
        "n_distinct_levels": int(len(set(values.tolist()))),
        "mean_by_turn": {int(t): float(np.mean(v)) for t, v in by_turn.items()},
        "pooled_within_turn_sd": pooled_within_sd,
        "erosion_snr": float(snr),
    }
