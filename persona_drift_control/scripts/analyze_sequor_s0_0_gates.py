#!/usr/bin/env python3
"""S0-0's three gates for the `constraint` line
(docs/experiments/constraint_signal_screening.md sections 3 and 8). CPU only.

    K1  readout range        -- does `y` vary at every turn, and do the 8
                                satisfaction patterns actually occur?
    K2  state beyond turn    -- does `y_t` predict the counterfactual gain
                                once the turn ramp is controlled? (the line's
                                existence question; ERGO died here at 97%
                                attenuation)
    K3  executor authority    -- is the mean one-step gain non-zero AND larger
                                than this design can resolve?

Every verdict is reported WITH the design's MDE in the same units as the
quantity tested. A CI crossing zero next to an MDE several times the expected
effect is not a negative result, it is an unpowered one -- ERGO's G-EKA-2 was
ruled undecidable on exactly that basis, and this script reports UNDECIDABLE
rather than letting a gate close a line it never had the power to test.

Gates are computed on the INDEPENDENT judge's readout. The self-judged readout
is optional and is used only to report how far the in-loop selection signal
sits from the reporting one, split by `u` -- the first (partial) look at plan
section 2's known risk that judge error correlates with the action.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import numpy as np

K_PATTERNS = 8
POWER_Z = 2.80  # two-sided alpha=0.05 at 80% power
N_BOOTSTRAP = 10000


def refuse_if_the_cap_bound(arm_report: dict, excluded: list[str]) -> dict:
    """Refuse to compute gates on an arm whose response cap bound on some item.

    Pre-registered after job 15756689 (18.6% of responses truncated, 75 of 87
    on two items whose constraints ask for length): a truncated response can
    fail a constraint the model would have satisfied, so `y` there is partly a
    measurement of our own token budget. Checked per item, since length is a
    function of the constraint set and a global average hid it.

    An item may be dropped, but never quietly: `--exclude-item` is the only way
    past this guard, and every exclusion is written into the gate report next
    to the number it changed. Excluding items BECAUSE their constraints demand
    length biases the item set toward constraints that are easy to keep, which
    is a finding about the design, not a detail.
    """

    by_item = arm_report.get("token_cap_by_item")
    if by_item is None:
        raise SystemExit(
            f"arm report has no per-item cap accounting: it predates the 2026-09-09 "
            f"criterion. Re-run the arm with the current runner rather than assuming."
        )
    criterion = arm_report.get("cap_criterion", 0.05)
    offending = {
        item: stats["token_cap_share"]
        for item, stats in by_item.items()
        if stats["token_cap_share"] > criterion and item not in excluded
    }
    if offending:
        raise SystemExit(
            f"the response cap bound on {len(offending)} item(s) above the {criterion:.0%} "
            f"criterion: " + ", ".join(f"{i} {s:.1%}" for i, s in offending.items()) +
            f". Their readout is partly a truncation measurement. Raise the cap and re-generate, "
            f"or pass --exclude-item for each (recorded in the report) if that is the ruling."
        )
    return {
        "cap_criterion": criterion,
        "excluded_items": sorted(excluded),
        "token_cap_share_overall": arm_report.get("token_cap_share"),
        "max_new_tokens": arm_report.get("max_new_tokens"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--independent-readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report with judge_kind=independent.")
    p.add_argument("--self-readout", type=pathlib.Path, default=None,
                   help="Optional judge_kind=self report over the same rows.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-item", action="append", default=[], metavar="ITEM_ID",
                   help="Drop this item from the pair set. The ONLY way past the response-cap "
                        "guard, and every exclusion is written into the report beside the numbers "
                        "it changed. Use it when the exclusion is the ruling, never to make a "
                        "criterion pass.")
    return p.parse_args()


def load_readout(path: pathlib.Path, expect_kind: str) -> dict:
    report = json.loads(path.read_text())
    if report["judge_kind"] != expect_kind:
        raise SystemExit(f"{path}: judge_kind={report['judge_kind']!r}, expected {expect_kind!r}")
    if report.get("mode") == "debug":
        raise SystemExit(f"{path}: arm mode=debug; debug artifacts are not evidence (plan section 8)")
    return report


def pairs_from(report: dict) -> list[dict]:
    """One record per counterfactual pair: the state before the action, the
    measured one-step gain, and the turn."""

    by_key = {(r["branch"], r["item_id"], r["turn"]): r for r in report["rows"]}
    out = []
    for (branch, item, turn), row in by_key.items():
        if branch != "reminded":
            continue
        base = by_key.get(("base", item, turn))
        prev = by_key.get(("base", item, turn - 1))
        if base is None or prev is None:
            continue
        if base["prefix_sha256"] != row["prefix_sha256"]:
            raise SystemExit(f"{item} turn {turn}: pair prefixes differ; delta is not a causal gain")
        if None in (row["y_graded"], base["y_graded"], prev["y_graded"]):
            continue
        out.append({
            "item_id": item, "turn": turn,
            "y_prev": prev["y_graded"], "y_base": base["y_graded"], "y_reminded": row["y_graded"],
            "delta_y": row["y_graded"] - base["y_graded"],
            "echo_jaccard_prev": row["echo_jaccard_prev"],
        })
    return sorted(out, key=lambda r: (r["item_id"], r["turn"]))


def _item_bootstrap(pairs: list[dict], statistic, seed: int) -> tuple[float, list[float], float]:
    """Resample ITEMS with replacement (a dialogue is the independent unit;
    turns within one are not). Returns (point, [lo, hi], bootstrap sd)."""

    rng = np.random.default_rng(seed)
    by_item: dict[str, list[dict]] = {}
    for pair in pairs:
        by_item.setdefault(pair["item_id"], []).append(pair)
    items = sorted(by_item)
    point = statistic(pairs)
    draws = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(len(items), size=len(items), replace=True)
        resampled = [p for idx in chosen for p in by_item[items[idx]]]
        value = statistic(resampled)
        if value is not None and np.isfinite(value):
            draws.append(value)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), [float(lo), float(hi)], float(np.std(draws, ddof=1))


def _ols(pairs: list[dict], regressors: tuple[str, ...]) -> np.ndarray | None:
    x = np.column_stack([np.ones(len(pairs))] + [[p[r] for p in pairs] for r in regressors])
    y = np.array([p["delta_y"] for p in pairs], dtype=float)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    return np.linalg.lstsq(x, y, rcond=None)[0]


def gate_k1(report: dict) -> dict:
    """Range: at every turn, cross-trajectory sd > 0 and >= 3 distinct values
    of `y`; and >= 4 of the 2^k satisfaction patterns occur in the arm.

    This is the gate the `defense` line failed after the fact rather than
    before (independent judge ceiling share 0.91, turn 1 sd = 0), which is why
    it runs first and closes the line on failure with no second judge attempt.
    """

    base = [r for r in report["rows"] if r["branch"] == "base" and r["y_graded"] is not None]
    by_turn: dict[int, list[float]] = {}
    patterns = set()
    for row in base:
        by_turn.setdefault(row["turn"], []).append(row["y_graded"])
        patterns.add(tuple(bool(f) for f in row["followed"]))
    per_turn = {
        turn: {
            "n": len(vals), "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "distinct": len(set(vals)), "ceiling_share": sum(1 for v in vals if v == 1.0) / len(vals),
        }
        for turn, vals in sorted(by_turn.items())
    }
    dead = [t for t, s in per_turn.items() if s["sd"] == 0.0 or s["distinct"] < 3]
    return {
        "criterion": "every turn: sd > 0 AND >= 3 distinct y; AND >= 4 of 8 satisfaction patterns",
        "per_turn": per_turn,
        "turns_failing": dead,
        "n_patterns_seen": len(patterns),
        "patterns_seen": sorted("".join("1" if b else "0" for b in p) for p in patterns),
        "overall_ceiling_share": sum(1 for r in base if r["y_graded"] == 1.0) / len(base),
        "pass": bool(not dead and len(patterns) >= 4),
    }


def gate_k2(pairs: list[dict], seed: int) -> dict:
    """State beyond the turn index: the `y_prev` coefficient in
    delta_y ~ y_prev + turn, with the attenuation against the marginal fit.

    Two degeneracies are checked BEFORE the verdict, because either one makes
    a CI that crosses zero uninformative:

    - a flat x-axis (`y_prev` with no spread): ERGO's G-EKA-2 had IQR 0 with
      75% of rows at one value, so there was nothing for a slope to be fitted
      against;
    - a slope MDE larger than the mean gain itself, i.e. the design could not
      have resolved a state effect even as big as the whole average effect.

    Either -> UNDECIDABLE (report and hand back), not a failed gate. A failed
    gate closes the line; an unpowered one closing it would be the mistake the
    pre-closure review caught on ERGO.
    """

    y_prev = [p["y_prev"] for p in pairs]
    q75, q25 = np.percentile(y_prev, [75, 25])
    mean_gain = float(np.mean([p["delta_y"] for p in pairs]))
    result = {
        "criterion": "y_prev coefficient CI excludes 0 AND attenuation vs the marginal fit < 50%",
        "y_prev_iqr": float(q75 - q25), "y_prev_distinct": len(set(y_prev)),
        "y_prev_distribution": {str(v): y_prev.count(v) for v in sorted(set(y_prev))},
        "mean_gain_for_scale": mean_gain,
        "slope_on_y_prev": None, "ci": None, "bootstrap_sd": None, "mde_at_80pct": None,
        "marginal_slope": None, "attenuation": None, "turn_coefficient": None,
        "degenerate_axis": False, "underpowered": None,
        "note": "UNDECIDABLE is not FAIL: a failed gate closes the line, an unpowered one "
                "would close it without having tested it (ERGO G-EKA-2 precedent).",
    }

    # The flat-axis check comes FIRST and is not a fit failure to be reported as
    # a null: a constant y_prev is collinear with the intercept, so there is no
    # slope to estimate and no CI to interpret (EK-A's `c_prev` IQR 0).
    full = _ols(pairs, ("y_prev", "turn"))
    marginal = _ols(pairs, ("y_prev",))
    result["degenerate_axis"] = bool((q75 - q25) == 0 or len(set(y_prev)) < 3
                                     or full is None or marginal is None)
    if result["degenerate_axis"]:
        result["verdict"] = "UNDECIDABLE"
        result["reason"] = ("y_prev has no usable spread (or the design matrix is rank "
                            "deficient): the regression has no x-axis")
        return result

    def slope(sample: list[dict]) -> float | None:
        beta = _ols(sample, ("y_prev", "turn"))
        return None if beta is None else float(beta[1])

    point, ci, sd = _item_bootstrap(pairs, slope, seed)
    mde = POWER_Z * sd
    attenuation = 1.0 - abs(point) / abs(marginal[1]) if marginal[1] != 0 else None
    underpowered = mde > abs(mean_gain) if mean_gain else True
    result.update({
        "slope_on_y_prev": point, "ci": ci, "bootstrap_sd": sd, "mde_at_80pct": mde,
        "marginal_slope": float(marginal[1]), "attenuation": attenuation,
        "turn_coefficient": float(full[2]), "underpowered": bool(underpowered),
    })
    if underpowered:
        result["verdict"] = "UNDECIDABLE"
        result["reason"] = (f"slope MDE {mde:.4f} exceeds the mean gain {abs(mean_gain):.4f}: this "
                            f"design could not have resolved a state effect even as large as the "
                            f"whole average effect")
    elif ci[0] * ci[1] > 0 and attenuation is not None and attenuation < 0.50:
        result["verdict"] = "PASS"
    else:
        result["verdict"] = "FAIL"
    return result


def gate_k3(pairs: list[dict], seed: int) -> dict:
    """Executor authority: mean(delta_y) CI excludes 0 AND the effect is at
    least this design's MDE. The second half is what a bare CI hides."""

    point, ci, sd = _item_bootstrap(pairs, lambda sample: float(np.mean([p["delta_y"] for p in sample])), seed)
    mde = POWER_Z * sd
    deltas = [p["delta_y"] for p in pairs]
    return {
        "criterion": "mean(delta_y) CI excludes 0 AND |effect| >= MDE",
        "mean_delta_y": point, "ci": ci, "bootstrap_sd": sd, "mde_at_80pct": mde,
        "effect_over_mde": abs(point) / mde if mde else None,
        "share_exactly_zero": sum(1 for d in deltas if d == 0) / len(deltas),
        "share_positive": sum(1 for d in deltas if d > 0) / len(deltas),
        "share_negative": sum(1 for d in deltas if d < 0) / len(deltas),
        "n_pairs": len(pairs), "n_items": len({p["item_id"] for p in pairs}),
        "pass": bool(ci[0] * ci[1] > 0 and abs(point) >= mde),
    }


def judge_agreement_by_action(independent: dict, self_report: dict) -> dict:
    """Plan section 2's known risk, first look: if the judge is more wrong on
    reminded rows (a reminder makes responses more formulaic), `B` is inflated
    by an instrument artifact rather than a causal gain. The gold set has no
    `u`, so this compares the two judges on the SAME rows instead -- weaker
    than a strong-judge sample, and it does not replace it."""

    self_by = {(r["branch"], r["item_id"], r["turn"]): r for r in self_report["rows"]}
    out = {}
    for label, want in (("u=0 (base)", 0), ("u=1 (reminded)", 1)):
        rows = [r for r in independent["rows"] if r["u_remind"] == want and r["y_graded"] is not None]
        both = [(r, self_by.get((r["branch"], r["item_id"], r["turn"]))) for r in rows]
        both = [(a, b) for a, b in both if b is not None and b["y_graded"] is not None]
        if not both:
            continue
        out[label] = {
            "n": len(both),
            "mean_y_independent": statistics.fmean(a["y_graded"] for a, _ in both),
            "mean_y_self": statistics.fmean(b["y_graded"] for _, b in both),
            "mean_signed_gap_self_minus_independent":
                statistics.fmean(b["y_graded"] - a["y_graded"] for a, b in both),
            "exact_agreement": sum(1 for a, b in both if a["y_graded"] == b["y_graded"]) / len(both),
        }
    return out


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    independent = load_readout(args.independent_readout, "independent")
    arm_report = json.loads((pathlib.Path(independent["arm_dir"]) / "arm_report.json").read_text())
    cap_record = refuse_if_the_cap_bound(arm_report, args.exclude_item)
    # Excluded items leave the readout entirely, not just the pair set: K1 reads
    # the rows directly, and an item dropped from K2/K3 while still propping up
    # K1's spread would be the worst of both.
    independent["rows"] = [r for r in independent["rows"] if r["item_id"] not in args.exclude_item]
    pairs = pairs_from(independent)
    if not pairs:
        raise SystemExit("no usable counterfactual pairs")

    k1 = gate_k1(independent)
    k2 = gate_k2(pairs, args.seed)
    k3 = gate_k3(pairs, args.seed)
    report = {
        "arm_dir": independent["arm_dir"], "agent_model": independent["agent_model"],
        "judge_model": independent["judge_model"], "judge_kind": independent["judge_kind"],
        "n_pairs": len(pairs), "n_items": len({p["item_id"] for p in pairs}),
        "response_cap": cap_record,
        "K1_readout_range": k1, "K2_state_beyond_turn": k2, "K3_executor_authority": k3,
    }
    if args.self_readout:
        report["self_vs_independent_by_action"] = judge_agreement_by_action(
            independent, load_readout(args.self_readout, "self")
        )

    print(f"{len(pairs)} pairs over {report['n_items']} items\n")
    print(f"K1 readout range      {'PASS' if k1['pass'] else 'FAIL'}  "
          f"(turns failing: {k1['turns_failing'] or 'none'}; patterns seen {k1['n_patterns_seen']}/8; "
          f"ceiling share {k1['overall_ceiling_share']:.4f})")
    print(f"K2 state beyond turn  {k2['verdict']}  slope {k2.get('slope_on_y_prev')} "
          f"CI {k2.get('ci')} MDE {k2.get('mde_at_80pct')}  attenuation {k2.get('attenuation')}")
    print(f"K3 executor authority {'PASS' if k3['pass'] else 'FAIL'}  mean(delta_y) "
          f"{k3['mean_delta_y']:+.4f} CI {k3['ci']} MDE {k3['mde_at_80pct']:.4f} "
          f"(effect/MDE {k3['effect_over_mde']:.2f}; {k3['share_exactly_zero']*100:.1f}% of pairs exactly 0)")
    if not k1["pass"]:
        print("\nK1 failed -> plan section 7 / screening section 5: CLOSE the line. "
              "The judge/readout chain is not to be repaired a second time.")
    elif k2["verdict"] == "FAIL":
        print("\nK2 failed -> the gain is a function of the turn index, not of the state: "
              "CLOSE the line (this is ERGO's death, at 97% attenuation).")
    elif k2["verdict"] == "UNDECIDABLE":
        print("\nK2 undecidable -> report and hand back. Do NOT close on an unpowered gate, "
              "and do NOT re-run the same design hoping for a narrower CI.")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
