#!/usr/bin/env python3
"""D1 (docs/experiments/readout_controllability_gate_plan.md 第一节):
is the `u_remind -> proj_pre_reply` input gain T3 Step 1 measured a persistent
state effect, or a same-turn artifact of the reminder text sitting inside the
context that gets projected?

Four checks, all on the same 736 transitions (920 rows, 10 arms: T3's
combined dataset) unless noted:

S0. Reproduce T3 Step 1 row C exactly (gate).
S1. Add u_{t+1} (the action taken the turn AFTER the one whose state is
    being predicted) -- corr(u_t, u_{t+1})=0.5010, so omitting it confounds
    the u_t coefficient with whatever u_{t+1} is doing.
S2. Restrict to arms whose action does not depend on the state (exogenous
    schedules), so u_t can't be confounded with an unobserved policy
    variable that reacts to x_t.
    S2a: phaseG_periodic alone (64 transitions), no arm FE.
    S2b: phaseG_periodic + phaseJ_fixed_t1 + phaseJ_fixed_t4 (192
    transitions), with arm FE. Reference-only: fixed_tk's u_t is
    1{turn==k}, near-collinear with the turn term.
S3. Persistence: does a reminder's effect survive one more turn once it is
    no longer the most recent action? u_{t-1} coefficient in
    x_{t+1} ~ 1 + x_t + u_{t-1} + u_t + u_{t+1} + turn + arm_FE.
S4. Causal-availability check (not a gate, a fact for the record):
    proj_pre_reply_t is itself a function of u_t (the reminder for THIS
    turn is already in the context that gets projected), so it cannot be
    the state the controller conditions on at decision time.

Reuses `group_by_key`/`ols` from analyze_readout_state.py unmodified (do not
edit that file -- its G0 gates are the baseline every earlier result is
checked against). CPU-only, no matplotlib.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from analyze_readout_state import group_by_key, ols  # noqa: E402

EXPANDED_READOUT = "outputs/koopman_case_study/refusal_direction_readout_expanded.json"
CACHED_READOUT = "outputs/koopman_case_study/refusal_direction_readout.json"
OUT_PATH = pathlib.Path("outputs/koopman_case_study/input_gain_robustness_report.json")


def key_fn(row: dict) -> str:
    return row["arm"] + "|" + row["trajectory_id"]


def consecutive_pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    """Adjacent-turn (a, b) pairs within one trajectory's rows (already
    turn-sorted by group_by_key)."""
    return [(a, b) for a, b in zip(rows, rows[1:]) if b["turn"] == a["turn"] + 1]


def consecutive_triples(rows: list[dict]) -> list[tuple[dict, dict, dict]]:
    """(t-1, t, t+1) triples: three consecutive turns, all present."""
    out = []
    for c, a, b in zip(rows, rows[1:], rows[2:]):
        if a["turn"] == c["turn"] + 1 and b["turn"] == a["turn"] + 1:
            out.append((c, a, b))
    return out


def arm_dummies(rows: list[dict]) -> tuple[list[str], list[str]]:
    arms_sorted = sorted({row["arm"] for row in rows})
    return arms_sorted[:-1], arms_sorted


def build_s0_s1(
    rows: list[dict], x_col: str, include_u_next: bool, include_arm_fe: bool
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    dummy_arms, arms_sorted = arm_dummies(rows) if include_arm_fe else ([], [])
    groups = group_by_key(rows, key_fn)
    X, y_next, n_reminded = [], [], 0
    for traj in groups.values():
        for a, b in consecutive_pairs(traj):
            row_x = [1.0, float(a[x_col]), float(a["u_remind"]), float(a["turn"])]
            if include_u_next:
                row_x.append(float(b["u_remind"]))
            if include_arm_fe:
                row_x.extend(1.0 if a["arm"] == arm else 0.0 for arm in dummy_arms)
            X.append(row_x)
            y_next.append(float(b[x_col]))
            if int(a["u_remind"]) != 0:
                n_reminded += 1
    return np.array(X), np.array(y_next), n_reminded, arms_sorted


def u_t_u_t1_corr(rows: list[dict]) -> float:
    groups = group_by_key(rows, key_fn)
    u_t, u_t1 = [], []
    for traj in groups.values():
        for a, b in consecutive_pairs(traj):
            u_t.append(float(a["u_remind"]))
            u_t1.append(float(b["u_remind"]))
    return float(np.corrcoef(u_t, u_t1)[0, 1])


def build_s3(
    rows: list[dict], x_col: str, include_arm_fe: bool
) -> tuple[np.ndarray, np.ndarray, int]:
    dummy_arms, _ = arm_dummies(rows) if include_arm_fe else ([], [])
    groups = group_by_key(rows, key_fn)
    X, y_next, n_triples = [], [], 0
    for traj in groups.values():
        for c, a, b in consecutive_triples(traj):
            row_x = [
                1.0,
                float(a[x_col]),
                float(c["u_remind"]),
                float(a["u_remind"]),
                float(b["u_remind"]),
                float(a["turn"]),
            ]
            if include_arm_fe:
                row_x.extend(1.0 if a["arm"] == arm else 0.0 for arm in dummy_arms)
            X.append(row_x)
            y_next.append(float(b[x_col]))
            n_triples += 1
    return np.array(X), np.array(y_next), n_triples


def build_s4(rows: list[dict], x_col: str, include_arm_fe: bool) -> tuple[np.ndarray, np.ndarray]:
    """Same-turn (contemporaneous) regression: x_t ~ 1 + u_t + turn + arm_FE,
    over every row (not transitions)."""
    dummy_arms, _ = arm_dummies(rows) if include_arm_fe else ([], [])
    X, x_t = [], []
    for row in rows:
        row_x = [1.0, float(row["u_remind"]), float(row["turn"])]
        if include_arm_fe:
            row_x.extend(1.0 if row["arm"] == arm else 0.0 for arm in dummy_arms)
        X.append(row_x)
        x_t.append(float(row[x_col]))
    return np.array(X), np.array(x_t)


def coef_report(gain: dict, idx: int) -> dict:
    return {"coef": gain["beta"][idx], "se": gain["se"][idx], "p": gain["p"][idx]}


def main() -> None:
    expanded_rows = json.loads(pathlib.Path(EXPANDED_READOUT).read_text())["rows"]
    cached_rows = json.loads(pathlib.Path(CACHED_READOUT).read_text())["rows"]
    combined_rows = expanded_rows + cached_rows

    report: dict = {"S0": {}, "S1": {}, "S2a": {}, "S2b": {}, "S3": {}, "S4": {}}

    # --- S0: reproduce T3 row C (736 transitions, 10 arms, arm FE) ---
    for col in ("proj_pre_reply", "proj_post_reply"):
        X, y, n_rem, arms = build_s0_s1(combined_rows, col, include_u_next=False, include_arm_fe=True)
        gain = ols(X, y)
        report["S0"][col] = {
            "u_t": coef_report(gain, 2),
            "r2": gain["r2"],
            "n_transitions": gain["n"],
            "n_reminded_transitions": n_rem,
            "arms": arms,
        }

    # --- S1: add u_{t+1} ---
    for col in ("proj_pre_reply", "proj_post_reply"):
        X, y, n_rem, arms = build_s0_s1(combined_rows, col, include_u_next=True, include_arm_fe=True)
        gain = ols(X, y)
        report["S1"][col] = {
            "u_t": coef_report(gain, 2),
            "u_t1": coef_report(gain, 4),
            "r2": gain["r2"],
            "n_transitions": gain["n"],
            "n_reminded_transitions": n_rem,
            "arms": arms,
        }
    report["S1"]["corr_u_t_u_t1"] = u_t_u_t1_corr(combined_rows)

    # --- S2a: phaseG_periodic alone, no arm FE, both specs ---
    periodic_rows = [r for r in combined_rows if r["arm"] == "phaseG_periodic"]
    for col in ("proj_pre_reply", "proj_post_reply"):
        entry = {}
        for spec, include_u_next in (("no_u_t1", False), ("with_u_t1", True)):
            X, y, n_rem, _ = build_s0_s1(periodic_rows, col, include_u_next=include_u_next, include_arm_fe=False)
            gain = ols(X, y)
            d = {"u_t": coef_report(gain, 2), "r2": gain["r2"], "n_transitions": gain["n"], "n_reminded_transitions": n_rem}
            if include_u_next:
                d["u_t1"] = coef_report(gain, 4)
                d["design_matrix_rank"] = int(np.linalg.matrix_rank(X))
                d["design_matrix_ncols"] = int(X.shape[1])
                d["design_matrix_cond"] = float(np.linalg.cond(X))
                d["identified"] = bool(np.linalg.matrix_rank(X) == X.shape[1])
            entry[spec] = d
        report["S2a"][col] = entry
    report["S2a"]["diagnostic_note"] = (
        "phaseG_periodic alternates u_remind deterministically by turn (0,1,0,1,0 for turns "
        "1-5, identical across every trajectory), so within this single-arm subset "
        "u_t + u_{t+1} == 1 for every transition. Including both in the same regression "
        "(no arm FE) makes the design matrix rank-deficient (see design_matrix_rank/ncols/cond "
        "under 'with_u_t1') -- the with_u_t1 coefficients above are a numerically unstable "
        "pseudo-inverse solution (p starts saturating at 1.0), not evidence for or against "
        "criterion 2. This is a structural identifiability failure of this exact single-arm "
        "spec, not a low-power null result."
    )

    # --- S2b: periodic + fixed_t1 + fixed_t4, with arm FE, both specs ---
    s2b_rows = [r for r in combined_rows if r["arm"] in ("phaseG_periodic", "phaseJ_fixed_t1", "phaseJ_fixed_t4")]
    for col in ("proj_pre_reply", "proj_post_reply"):
        entry = {}
        for spec, include_u_next in (("no_u_t1", False), ("with_u_t1", True)):
            X, y, n_rem, arms = build_s0_s1(s2b_rows, col, include_u_next=include_u_next, include_arm_fe=True)
            gain = ols(X, y)
            d = {
                "u_t": coef_report(gain, 2),
                "r2": gain["r2"],
                "n_transitions": gain["n"],
                "n_reminded_transitions": n_rem,
                "arms": arms,
            }
            if include_u_next:
                d["u_t1"] = coef_report(gain, 4)
            entry[spec] = d
        report["S2b"][col] = entry
    report["S2b"]["note"] = "reference only -- fixed_tk's u_t is 1{turn==k}, near-collinear with the turn term"

    # --- S3: persistence, (t-1, t, t+1) triples, arm FE ---
    for col in ("proj_pre_reply", "proj_post_reply"):
        X, y, n_triples = build_s3(combined_rows, col, include_arm_fe=True)
        gain = ols(X, y)
        report["S3"][col] = {
            "u_t_minus_1": coef_report(gain, 2),
            "u_t": coef_report(gain, 3),
            "u_t1": coef_report(gain, 4),
            "r2": gain["r2"],
            "n_triples": n_triples,
        }

    # --- S4: contemporaneous (causal-availability) check, all rows, arm FE ---
    for col in ("proj_pre_reply", "proj_post_reply"):
        X, x_t = build_s4(combined_rows, col, include_arm_fe=True)
        gain = ols(X, x_t)
        report["S4"][col] = {"u_t": coef_report(gain, 1), "r2": gain["r2"], "n_rows": gain["n"]}
    report["S4"]["conclusion"] = (
        "proj_pre_reply is a function of the same-turn action; not observable at decision time"
    )

    # --- RC-A preregistered verdict ---
    s1_pre = report["S1"]["proj_pre_reply"]
    rc_a_1 = s1_pre["u_t"]["coef"] > 0 and s1_pre["u_t"]["p"] < 0.05
    s2a_with = report["S2a"]["proj_pre_reply"]["with_u_t1"]
    rc_a_2_literal = s2a_with["u_t"]["coef"] > 0 and s2a_with["u_t"]["p"] < 0.05
    s2b_with = report["S2b"]["proj_pre_reply"]["with_u_t1"]
    rc_a_2_reference = s2b_with["u_t"]["coef"] > 0 and s2b_with["u_t"]["p"] < 0.05
    s3_pre = report["S3"]["proj_pre_reply"]
    s3_post = report["S3"]["proj_post_reply"]
    rc_a_3 = (s3_pre["u_t_minus_1"]["coef"] > 0 and s3_pre["u_t_minus_1"]["p"] < 0.05) or (
        s3_post["u_t_minus_1"]["coef"] > 0 and s3_post["u_t_minus_1"]["p"] < 0.05
    )
    rc_a = {
        "criterion_1_S1_u_t_positive_significant": rc_a_1,
        "criterion_2_S2a_periodic_u_t_positive_significant_LITERAL_SPEC": rc_a_2_literal,
        "criterion_2_note": (
            "the literal S2a with_u_t1 spec is rank-deficient for phaseG_periodic alone "
            "(see S2a.diagnostic_note) so this literal value is not meaningful evidence either "
            "way; criterion_2_S2a_..._reference_S2b uses the well-identified cross-arm S2b "
            "with_u_t1 spec (periodic+fixed_t1+fixed_t4, arm FE, cond~182) as the best "
            "available substitute, reference-only per the plan's own caveat on S2b"
        ),
        "criterion_2_S2a_periodic_u_t_positive_significant_reference_S2b": rc_a_2_reference,
        "criterion_3_S3_u_t_minus_1_positive_significant_either_channel": rc_a_3,
        "all_pass_literal_spec": bool(rc_a_1 and rc_a_2_literal and rc_a_3),
        "all_pass_using_S2b_reference_for_criterion_2": bool(rc_a_1 and rc_a_2_reference and rc_a_3),
    }
    report["RC_A_verdict"] = rc_a

    print("=== D1: input-gain robustness (S0-S4) ===")
    print("\nS0 (reproduce T3 row C, 736 transitions, 10 arms, arm FE):")
    for col, r in report["S0"].items():
        print(f"  {col:<16} u_t={r['u_t']['coef']:+.4f} (se={r['u_t']['se']:.4f}, p={r['u_t']['p']:.4e}) R2={r['r2']:.3f} n={r['n_transitions']}")

    print("\nS1 (add u_t+1):")
    for col, r in report["S1"].items():
        if col == "corr_u_t_u_t1":
            continue
        print(
            f"  {col:<16} u_t={r['u_t']['coef']:+.4f} (p={r['u_t']['p']:.4e})  "
            f"u_t+1={r['u_t1']['coef']:+.4f} (p={r['u_t1']['p']:.4e})  R2={r['r2']:.3f}"
        )
    print(f"  corr(u_t, u_t+1) = {report['S1']['corr_u_t_u_t1']:.4f}")

    print("\nS2a (phaseG_periodic alone, 64 transitions, no arm FE):")
    for col, entry in report["S2a"].items():
        if col == "diagnostic_note":
            continue
        for spec, r in entry.items():
            extra = f" u_t+1={r['u_t1']['coef']:+.4f}(p={r['u_t1']['p']:.4e})" if "u_t1" in r else ""
            print(f"  {col:<16}{spec:<12} u_t={r['u_t']['coef']:+.4f}(p={r['u_t']['p']:.4e}) n={r['n_transitions']}{extra}")

    print("\nS2b (periodic+fixed_t1+fixed_t4, 192 transitions, arm FE, reference only):")
    for col, entry in report["S2b"].items():
        if col == "note":
            continue
        for spec, r in entry.items():
            extra = f" u_t+1={r['u_t1']['coef']:+.4f}(p={r['u_t1']['p']:.4e})" if "u_t1" in r else ""
            print(f"  {col:<16}{spec:<12} u_t={r['u_t']['coef']:+.4f}(p={r['u_t']['p']:.4e}) n={r['n_transitions']}{extra}")

    print("\nS3 (persistence, (t-1,t,t+1) triples, arm FE):")
    for col, r in report["S3"].items():
        print(
            f"  {col:<16} u_t-1={r['u_t_minus_1']['coef']:+.4f}(p={r['u_t_minus_1']['p']:.4e})  "
            f"u_t={r['u_t']['coef']:+.4f}(p={r['u_t']['p']:.4e})  u_t+1={r['u_t1']['coef']:+.4f}(p={r['u_t1']['p']:.4e})  n={r['n_triples']}"
        )

    print("\nS4 (contemporaneous / causal-availability check, not a gate):")
    for col, r in report["S4"].items():
        if col == "conclusion":
            continue
        print(f"  {col:<16} u_t={r['u_t']['coef']:+.4f}(p={r['u_t']['p']:.4e}) n={r['n_rows']}")
    print(f"  -> {report['S4']['conclusion']}")

    print("\n=== RC-A preregistered verdict ===")
    for k, v in rc_a.items():
        print(f"  {k}: {v}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
