#!/usr/bin/env python3
"""Self-judge vs independent-judge comparison for the defense line
(CPU-only, seconds) -- the counterpart of scripts/compare_judge_runs.py,
which did this for the sycophancy line's three-class label.

Input: each arm's original `trajectories.jsonl` (scored by the self-judge,
agent == judge == Qwen/Qwen3-4B) and the rejudged copy under
`<arm>/<--rejudge-subdir>/trajectories.jsonl` produced by
scripts/rejudge_safety_runs.py. The pairing is exact by construction: the
rejudge replays the judge call on the stored `agent_message`, so the two
score sets describe the same replies and the same controller decisions.

Four questions, in the order they change what may be believed:

1. Direction and size of the bias (confusion matrix, sign test, mean shift).
   The sycophancy line found one-way misses only.
2. Is it a constant level offset or does it grow with turn? A level offset
   compresses new-Q1's effect size without faking its direction; a
   turn-dependent one distorts the slope itself.
3. Does the bias depend on `u_remind`? This is the one that can change a
   Phase A-J conclusion rather than just its error bars -- a self-judge that
   rates its own post-reminder replies generously would manufacture part of
   the measured defense effect. A constant offset cancels out of a
   between-arm difference; a reminder-dependent one does not.
4. What actually changes in the arm-level numbers: late_mean_y, the new-Q1
   test each phase was judged on, the ceiling share / erosion SNR that
   docs/experiments/koopman_defense_pilot.md section 五 uses to argue the
   readout is the binding constraint, the ranking of the arms, and the
   adaptive-vs-best-fixed paired bootstrap that is Phase J's headline.

What this cannot answer: for the reactive arms (`threshold`,
`koopman_mpc*`) the controller's decisions were taken from the self-judge's
y_probe. Rejudging re-measures those trajectories; it does not re-run them.
If the bias turns out to matter, the reactive arms need a real rerun with
`judge_model` set -- the fixed-schedule arms do not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# Reuse (not re-derive) the arm list and the two metric helpers Phase J's
# headline numbers are computed with, so a difference between this script's
# output and analyze_budget_arm_comparison.py's can only come from the
# scores, never from a second implementation of late_mean_y or of the
# bootstrap.
from analyze_budget_arm_comparison import (  # noqa: E402
    DEFAULT_ARMS,
    _paired_bootstrap,
    _per_trajectory_metrics,
)

from persona_drift.analysis_adversarial import analyze_adversarial_screening  # noqa: E402
from persona_drift.judge_bias import analyze_judge_bias, pair_rows, readout_quality  # noqa: E402
from persona_drift.modeling.dataset import load_trajectories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="repeatable; overrides analyze_budget_arm_comparison.DEFAULT_ARMS entirely when given",
    )
    parser.add_argument("--rejudge-subdir", default="rejudge_qwen3_4b_instruct_2507")
    parser.add_argument("--adaptive-arm", default="koopman_budget1")
    parser.add_argument("--fixed-arm-prefix", default="fixed_t")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/judge_bias_comparison.json")
    )
    return parser.parse_args()


def _arm_metrics(rows: list[dict]) -> dict:
    per_traj = _per_trajectory_metrics(rows)
    report = analyze_adversarial_screening(rows)
    q1 = report["new_q1_escalation"]
    q3 = report["new_q3_autocorrelation"]
    return {
        "n_trajectories": len(per_traj),
        "late_mean_y": float(np.mean([m["late_mean_y"] for m in per_traj.values()])),
        "terminal_y_mean": float(np.mean([m["terminal_y"] for m in per_traj.values()])),
        "mean_y": float(np.mean([m["mean_y"] for m in per_traj.values()])),
        "new_q1_mean_slope": float(np.mean(list(q1["per_attack_mean_slope"].values()))),
        "new_q1_p": q1["t_test_mean_slope_vs_zero"]["p"],
        "new_q1_pass": q1["pass"],
        "new_q3_r": q3["r"],
        "new_q3_slope": q3["slope"],
        "readout": readout_quality(rows),
        "per_trajectory": per_traj,
    }


def main() -> None:
    args = parse_args()
    arms = dict(pair.split("=", 1) for pair in args.arm) if args.arm else DEFAULT_ARMS

    per_arm: dict[str, dict] = {}
    all_pairs: list[dict] = []
    for name, directory in arms.items():
        self_path = pathlib.Path(directory) / "trajectories.jsonl"
        indep_path = pathlib.Path(directory) / args.rejudge_subdir / "trajectories.jsonl"
        if not self_path.exists() or not indep_path.exists():
            print(f"!! {name}: missing {'self' if not self_path.exists() else 'rejudged'} scores, skipped")
            continue
        self_rows, indep_rows = load_trajectories(self_path), load_trajectories(indep_path)
        pairs, pairing = pair_rows(self_rows, indep_rows)
        for pair in pairs:
            all_pairs.append({**pair, "arm": name})
        per_arm[name] = {
            "pairing": pairing,
            "bias": analyze_judge_bias(pairs),
            "self_judge": _arm_metrics(self_rows),
            "independent_judge": _arm_metrics(indep_rows),
        }

    if not per_arm:
        raise SystemExit(f"no arm has both scores -- run scripts/rejudge_safety_runs.py first (looking for */{args.rejudge_subdir}/trajectories.jsonl)")

    pooled = analyze_judge_bias(all_pairs)
    pairing_ok = all(a["pairing"]["n_text_mismatch"] == 0 for a in per_arm.values())

    print("=" * 100)
    print("PAIRING (must be exact: the rejudge replays the judge on the stored agent_message)")
    for name, arm in per_arm.items():
        p = arm["pairing"]
        print(
            f"  {name:<32} paired={p['n_paired']:>4}  text_mismatch={p['n_text_mismatch']}  "
            f"self_only={p['n_self_only']}  indep_only={p['n_indep_only']}  "
            f"nan(self/indep)={p['n_nan_self']}/{p['n_nan_indep']}"
        )
    if not pairing_ok:
        print("  !! text mismatches present -- the numbers below are NOT a paired comparison, stop and check")

    o = pooled["overall"]
    print("\n" + "=" * 100)
    print(f"1. BIAS, pooled over arms (n={o['n']} rows)")
    print(f"  mean y_safety: self={o['mean_self']:.4f}  independent={o['mean_indep']:.4f}  "
          f"diff={o['mean_diff_indep_minus_self']:+.4f}")
    print(f"  disagreements: {o['n_disagree']} ({o['disagree_rate']:.1%})  "
          f"independent stricter={o['n_independent_stricter']}  looser={o['n_independent_looser']}  "
          f"sign test p={o['sign_test_p']:.3g}")
    print("  confusion (self -> independent), top cells:")
    for cell, n in sorted(pooled["confusion"].items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {cell}: {n}")

    print("\n2. IS IT A LEVEL SHIFT OR TURN-DEPENDENT?")
    for turn, s in pooled["by_turn"].items():
        print(f"  turn {turn}: self={s['mean_self']:.4f}  indep={s['mean_indep']:.4f}  "
              f"diff={s['mean_diff_indep_minus_self']:+.4f}  disagree={s['disagree_rate']:.1%} (n={s['n']})")
    b = pooled["bias_vs_turn"]
    if b:
        print(f"  signed-diff vs turn: slope={b['signed_diff_slope']:+.4f} p={b['signed_diff_p']:.4f}  "
              f"| disagreement-rate vs turn: slope={b['disagreement_rate_slope']:+.4f} p={b['disagreement_rate_p']:.4f}")
        print("  (p>=0.05 on the signed diff => approximately a constant level offset: it compresses the")
        print("   effect size but does not fabricate or reverse the erosion trend. p<0.05 => new-Q1's slope")
        print("   itself is biased and every phase's slope needs re-reading, not just its error bars.)")

    c = pooled["concentration"]
    print(f"\n   concentration: {c['n_disagreeing_rows']} disagreeing rows over "
          f"{c['n_attacks_with_any_disagreement']}/{c['n_attacks_total']} attacks, "
          f"{c['n_trajectories_with_any_disagreement']}/{c['n_trajectories_total']} trajectories; "
          f"largest single attack = {c['max_single_attack_share']:.1%} of them")
    print(f"   top attacks: {c['top_attacks']}")

    print("\n3. DOES THE BIAS DEPEND ON u_remind? (the one that can change an arm comparison, not just its CI)")
    for label in ("unreminded", "reminded"):
        s = pooled["by_u_remind"][label]
        if s.get("n"):
            print(f"  {label:<11} n={s['n']:>4}  self={s['mean_self']:.4f}  indep={s['mean_indep']:.4f}  "
                  f"diff={s['mean_diff_indep_minus_self']:+.4f}  disagree={s['disagree_rate']:.1%}")
    d = pooled["by_u_remind"].get("difference_of_biases")
    if d:
        print(f"  reminded minus unreminded bias = {d['mean_reminded_minus_unreminded']:+.4f} "
              f"(Welch t={d['welch_t']:.3f}, p={d['welch_p']:.4f})")
        print("  (p>=0.05 => the bias is reminder-independent and cancels out of arm differences.)")

    print("\n" + "=" * 100)
    print("4. WHAT CHANGES IN THE ARM-LEVEL NUMBERS")
    header = f"{'arm':<32}{'late_y self':>12}{'late_y ind':>12}{'Δ':>9}{'q1p self':>10}{'q1p ind':>9}{'ceil self':>11}{'ceil ind':>10}{'snr self':>10}{'snr ind':>9}"
    print(header)
    for name, arm in per_arm.items():
        s, i = arm["self_judge"], arm["independent_judge"]
        print(
            f"{name:<32}{s['late_mean_y']:>12.4f}{i['late_mean_y']:>12.4f}"
            f"{i['late_mean_y'] - s['late_mean_y']:>+9.4f}{s['new_q1_p']:>10.4f}{i['new_q1_p']:>9.4f}"
            f"{s['readout']['ceiling_share']:>11.3f}{i['readout']['ceiling_share']:>10.3f}"
            f"{s['readout']['erosion_snr']:>10.2f}{i['readout']['erosion_snr']:>9.2f}"
        )
    print("  (ceil = share of rows pinned at y_safety=1.00; snr = |turn-1 mean - last-turn mean| / pooled")
    print("   within-turn sd, the same definition koopman_defense_pilot.md section 五 reports as judge 1.63")
    print("   vs activation projection 4.20 -- if the independent judge raises it, the readout's resolution")
    print("   problem is partly the self-judging, not the 5-level scale.)")

    ranking = {}
    for judge_key in ("self_judge", "independent_judge"):
        ranking[judge_key] = sorted(per_arm, key=lambda n: -per_arm[n][judge_key]["late_mean_y"])
    print(f"\n  arm ranking by late_mean_y, self-judge:        {' > '.join(ranking['self_judge'])}")
    print(f"  arm ranking by late_mean_y, independent judge: {' > '.join(ranking['independent_judge'])}")
    rank_rho = float("nan")
    if len(per_arm) >= 3:
        names = list(per_arm)
        rank_rho = float(
            stats.spearmanr(
                [per_arm[n]["self_judge"]["late_mean_y"] for n in names],
                [per_arm[n]["independent_judge"]["late_mean_y"] for n in names],
            ).statistic
        )
        print(f"  Spearman rho between the two orderings: {rank_rho:.4f}")

    comparisons: dict[str, dict] = {}
    fixed_arms = [n for n in per_arm if n.startswith(args.fixed_arm_prefix)]
    if args.adaptive_arm in per_arm and fixed_arms:
        print(f"\n  Phase J headline under each judge ({args.adaptive_arm} vs its best fixed allocation,")
        print("  paired per trajectory, 95% bootstrap CI; the best fixed arm is picked on the same data,")
        print("  which biases against the adaptive arm, exactly as analyze_budget_arm_comparison.py does it):")
        for judge_key in ("self_judge", "independent_judge"):
            best = max(fixed_arms, key=lambda n: per_arm[n][judge_key]["late_mean_y"])
            adaptive = per_arm[args.adaptive_arm][judge_key]["per_trajectory"]
            fixed = per_arm[best][judge_key]["per_trajectory"]
            shared = sorted(set(adaptive) & set(fixed))
            if not shared:
                continue
            diffs = np.array([adaptive[t]["late_mean_y"] - fixed[t]["late_mean_y"] for t in shared])
            boot = _paired_bootstrap(diffs, args.bootstrap, args.bootstrap_seed)
            comparisons[judge_key] = {"best_fixed_arm": best, **boot}
            print(
                f"    {judge_key:<19} vs {best}: mean_diff={boot['mean_diff']:+.4f} "
                f"95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] (n={boot['n_pairs']})"
            )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rejudge_subdir": args.rejudge_subdir,
        "pairing_exact": pairing_ok,
        "pooled_bias": pooled,
        "arms": {
            name: {
                key: (
                    {k: v for k, v in arm[key].items() if k != "per_trajectory"}
                    if key in ("self_judge", "independent_judge")
                    else arm[key]
                )
                for key in arm
            }
            for name, arm in per_arm.items()
        },
        "ranking": ranking,
        "ranking_spearman_rho": rank_rho,
        "adaptive_vs_best_fixed": comparisons,
    }
    args.out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
