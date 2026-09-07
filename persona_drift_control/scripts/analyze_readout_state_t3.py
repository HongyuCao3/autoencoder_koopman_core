#!/usr/bin/env python3
"""docs/experiments/adaptive_vs_fixed_claim_plan.md T3 Step 1 (section 5.1.1):
does the actuator (`u_remind`) move the activation-projection readout once
you control for which arm a transition came from?

Reuses `scripts/analyze_readout_state.py`'s `ols` (plain OLS, classical SEs)
and `build_input_gain_transitions` (the no-arm-fixed-effects control, row B)
unmodified; adds an arm-fixed-effects variant (rows A and C) here, since
`analyze_readout_state.py`'s own gates (G0-1/G0-2/G0-3) must stay untouched
by this extension.

Three regression rows, `proj_pre_reply` and `proj_post_reply` each:
  A. new 6-arm data (600 rows, T3's own expanded readout), WITH arm FE.
  B. same 600 rows, WITHOUT arm FE (control -- exposes whether the naive
     model was just fitting "which arm", not "was this transition
     reminded").
  C. the 600 new rows + the cached 4 Phase J arms (920 rows total), WITH
     arm FE -- this is the row the plan's pre-registered judgment is based
     on (proj_pre_reply only).

CPU-only. Run directly (no sbatch) once
outputs/koopman_case_study/refusal_direction_readout_expanded.json exists.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from analyze_readout_state import build_input_gain_transitions, group_by_key, ols  # noqa: E402

EXPANDED_READOUT = "outputs/koopman_case_study/refusal_direction_readout_expanded.json"
CACHED_READOUT = "outputs/koopman_case_study/refusal_direction_readout.json"
OUT_PATH = pathlib.Path("outputs/koopman_case_study/readout_state_report_t3.json")


def key_fn(row: dict) -> str:
    return row["arm"] + "|" + row["trajectory_id"]


def build_input_gain_transitions_with_arm_fe(rows: list[dict], x_col: str) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Same transitions as `build_input_gain_transitions`, plus one-hot arm
    dummies appended after [1, x_t, u_remind_t, turn] (one column dropped to
    avoid collinearity with the intercept).
    """
    groups = group_by_key(rows, key_fn)
    arms_sorted = sorted({row["arm"] for row in rows})
    reference_arm = arms_sorted[-1]
    dummy_arms = arms_sorted[:-1]

    X, y_next, n_reminded = [], [], 0
    for traj in groups.values():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            base = [1.0, float(a[x_col]), float(a["u_remind"]), float(a["turn"])]
            dummies = [1.0 if a["arm"] == arm else 0.0 for arm in dummy_arms]
            X.append(base + dummies)
            y_next.append(float(b[x_col]))
            if int(a["u_remind"]) != 0:
                n_reminded += 1
    return np.array(X), np.array(y_next), n_reminded, arms_sorted


def summarize_gain(gain: dict, n_reminded: int) -> dict:
    return {
        "u_remind_coef": gain["beta"][2],
        "u_remind_se": gain["se"][2],
        "u_remind_p": gain["p"][2],
        "r2": gain["r2"],
        "n_transitions": gain["n"],
        "n_reminded_transitions": n_reminded,
    }


def main() -> None:
    expanded_rows = json.loads(pathlib.Path(EXPANDED_READOUT).read_text())["rows"]
    cached_rows = json.loads(pathlib.Path(CACHED_READOUT).read_text())["rows"]
    combined_rows = expanded_rows + cached_rows

    report = {"row_A_expanded_with_arm_fe": {}, "row_B_expanded_no_arm_fe": {}, "row_C_combined_with_arm_fe": {}}

    for col in ("proj_pre_reply", "proj_post_reply"):
        X, y, n_rem, arms = build_input_gain_transitions_with_arm_fe(expanded_rows, col)
        gain = ols(X, y)
        report["row_A_expanded_with_arm_fe"][col] = {**summarize_gain(gain, n_rem), "arms": arms}

        X, y, n_rem = build_input_gain_transitions(expanded_rows, key_fn, col)
        gain = ols(X, y)
        report["row_B_expanded_no_arm_fe"][col] = summarize_gain(gain, n_rem)

        X, y, n_rem, arms = build_input_gain_transitions_with_arm_fe(combined_rows, col)
        gain = ols(X, y)
        report["row_C_combined_with_arm_fe"][col] = {**summarize_gain(gain, n_rem), "arms": arms}

    print(f"{'row':<28}{'readout':<16}{'u_remind coef':>15}{'se':>10}{'p':>10}{'R2':>8}{'reminded/total':>16}")
    for row_name, cols in report.items():
        for col, r in cols.items():
            print(
                f"{row_name:<28}{col:<16}{r['u_remind_coef']:>15.4f}{r['u_remind_se']:>10.4f}"
                f"{r['u_remind_p']:>10.4f}{r['r2']:>8.3f}"
                f"{r['n_reminded_transitions']:>8d}/{r['n_transitions']:<7d}"
            )

    row_a = report["row_A_expanded_with_arm_fe"]["proj_pre_reply"]
    row_b = report["row_B_expanded_no_arm_fe"]["proj_pre_reply"]
    sign_a = row_a["u_remind_coef"] >= 0
    sign_b = row_b["u_remind_coef"] >= 0
    sig_a = row_a["u_remind_p"] < 0.05
    sig_b = row_b["u_remind_p"] < 0.05
    if sign_a != sign_b or sig_a != sig_b:
        print(
            "\n!! confounding evidence: row A (with arm FE) and row B (without) disagree in sign "
            "and/or significance side for proj_pre_reply -- report this regardless of the final verdict."
        )

    row_c = report["row_C_combined_with_arm_fe"]["proj_pre_reply"]
    verdict = "continue" if (row_c["u_remind_p"] < 0.05 and row_c["u_remind_coef"] > 0) else "terminate"
    report["preregistered_judgment"] = {
        "basis": "row_C_combined_with_arm_fe.proj_pre_reply",
        "u_remind_coef": row_c["u_remind_coef"],
        "u_remind_p": row_c["u_remind_p"],
        "verdict": verdict,
    }
    print(f"\npreregistered judgment (row C, proj_pre_reply): coef={row_c['u_remind_coef']:+.4f} p={row_c['u_remind_p']:.4f} -> {verdict}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
