#!/usr/bin/env python3
"""Fixes the diagnostics of docs/experiments/adaptive_vs_fixed_claim_plan.md
section 0 into a re-runnable script (that section's numbers are this
script's expected output -- see the plan's T0).

Three segments, all offline / CPU-only (no GPU, no matplotlib):

1. Variance decomposition + de-turned lag-1 autocorrelation on `y_safety`,
   self vs. independent judge, on screening and Phase E zero_control. Answers
   "does the readout carry any state beyond the common turn trend".
2. The same decomposition on the cached activation-projection readout
   (`refusal_direction_readout.json`), plus an input-gain OLS
   (`x_{t+1} ~ 1 + x_t + u_remind_t + turn`) testing whether the actuator can
   move that state.
3. The two claim gates on the 16 held-out-attack x seeds{0,1} trajectories
   that exist in all 8 Phase J budget-1 arms: gate 1 is adaptive vs.
   zero_control; gate 2 is adaptive vs. an equal-cost random-mix baseline
   built from zero_control + each fixed schedule at koopman's own spend rate.

Every number this script prints is checked against
docs/experiments/adaptive_vs_fixed_claim_plan.md section 0's tables (that
section IS this script's spec) -- gates G0-1/G0-2/G0-3 in that doc.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from persona_drift.modeling.dataset import load_trajectories  # noqa: E402

OUT_PATH = pathlib.Path("outputs/koopman_case_study/readout_state_report.json")

LATE_WINDOW = (3, 4, 5)
TERMINAL_TURN = 5
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0

# Section 1.2's three (dataset, judge) rows.
SEGMENT1_DATASETS = {
    "screening_self": "outputs/adversarial_screening/trajectories.jsonl",
    "zero_control_self": "outputs/koopman_defense_phaseE_zero_control/trajectories.jsonl",
    "zero_control_indep": "outputs/koopman_defense_phaseE_zero_control/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl",
}

# Section 1.2 segment 2's cached activation-projection readout.
REFUSAL_DIRECTION_READOUT = "outputs/koopman_case_study/refusal_direction_readout.json"

# Section 1.2 segment 3's judge-value-rule table: for each arm, where the
# self-judged and independent-judge trajectories.jsonl live. Reactive arms
# (koopman_b1, threshold_b1) MUST use their own real `_indepjudge` run
# directory, not an offline rejudge of the self-judged run -- offline
# rejudge cannot change a decision the controller already made on the
# self-judged score (failure mode 4 in the plan's section 8).
ARM_SOURCES = {
    "zero_control": (
        "outputs/koopman_defense_phaseE_zero_control/trajectories.jsonl",
        "outputs/koopman_defense_phaseE_zero_control/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl",
    ),
    "koopman_b1": (
        "outputs/koopman_defense_phaseJ_budget1_koopman/trajectories.jsonl",
        "outputs/koopman_defense_phaseJ_budget1_koopman_indepjudge/trajectories.jsonl",
    ),
    "threshold_b1": (
        "outputs/koopman_defense_phaseJ_budget1_threshold/trajectories.jsonl",
        "outputs/koopman_defense_phaseJ_budget1_threshold_indepjudge/trajectories.jsonl",
    ),
    **{
        f"fixed_t{k}": (
            f"outputs/koopman_defense_phaseJ_budget1_fixed_t{k}/trajectories.jsonl",
            f"outputs/koopman_defense_phaseJ_budget1_fixed_t{k}/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl",
        )
        for k in range(1, 6)
    },
}
ADAPTIVE_ARM = "koopman_b1"
FIXED_ARM_PREFIX = "fixed_t"

# Section 1.3's stopword list -- copied verbatim, do not extend.
STOPWORDS = set(
    """
a an the of to in on for and or with without how what who was were is are be been being
by at from as that this these those it its his her their you your i we they can could would should
do does did done make making made use used using give given get getting about into over under more
most some any all not no if then than there here when where which while up out off very just also
me my him them he she
""".split()
)


# ---------------------------------------------------------------------------
# Segment 1: variance decomposition + de-turned lag-1 autocorrelation.
# ---------------------------------------------------------------------------


def group_by_key(rows: list[dict], key_fn) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    for traj_rows in groups.values():
        traj_rows.sort(key=lambda r: r["turn"])
    return groups


def variance_decomposition(rows: list[dict], y_col: str, key_fn=lambda r: r["trajectory_id"]) -> dict:
    """docs/experiments/adaptive_vs_fixed_claim_plan.md section 1.2 segment 1.

    Pools rows (not trajectories) to build the per-turn mean, de-means every
    row by its own turn's mean, then computes lag-1 autocorrelation on
    adjacent-turn pairs within each trajectory (raw = on `y` itself, demeaned
    = on the turn-residual) and the share of residual variance explained by
    a stable per-trajectory offset.
    """
    y = np.array([float(row[y_col]) for row in rows])
    turns = np.array([int(row["turn"]) for row in rows])

    turn_mean = {t: float(y[turns == t].mean()) for t in sorted(set(turns.tolist()))}
    turn_mean_bcast = np.array([turn_mean[t] for t in turns])
    resid = y - turn_mean_bcast
    var_by_turn = float(np.var(turn_mean_bcast) / np.var(y))

    index_of = {(key_fn(row), row["turn"]): i for i, row in enumerate(rows)}
    groups = group_by_key(rows, key_fn)

    raw_t, raw_t1, dem_t, dem_t1 = [], [], [], []
    for key, traj in groups.items():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            ia, ib = index_of[(key, a["turn"])], index_of[(key, b["turn"])]
            raw_t.append(y[ia])
            raw_t1.append(y[ib])
            dem_t.append(resid[ia])
            dem_t1.append(resid[ib])

    lag1_raw, p_raw = stats.pearsonr(raw_t, raw_t1) if len(raw_t) > 1 else (float("nan"), float("nan"))
    lag1_demeaned, p_demeaned = (
        stats.pearsonr(dem_t, dem_t1) if len(dem_t) > 1 else (float("nan"), float("nan"))
    )

    traj_mean_resid = {key: float(np.mean([resid[index_of[(key, r["turn"])]] for r in traj])) for key, traj in groups.items()}
    traj_bcast = np.array([traj_mean_resid[key_fn(row)] for row in rows])
    stable_traj_share = float(np.var(traj_bcast) / np.var(resid))

    return {
        "n_rows": len(rows),
        "n_trajectories": len(groups),
        "n_lag1_pairs": len(raw_t),
        "var_by_turn": var_by_turn,
        "lag1_raw": float(lag1_raw),
        "lag1_raw_p": float(p_raw),
        "lag1_demeaned": float(lag1_demeaned),
        "lag1_demeaned_p": float(p_demeaned),
        "stable_traj_share": stable_traj_share,
    }


def run_segment1() -> dict:
    report = {}
    for label, path in SEGMENT1_DATASETS.items():
        rows = load_trajectories(path)
        report[label] = {"path": path, **variance_decomposition(rows, y_col="y_safety")}
    return report


def print_segment1(report: dict) -> None:
    print("\n=== segment 1: y_safety variance decomposition (self vs. independent judge) ===")
    print(f"{'dataset':<20}{'n_rows':>7}{'var_by_turn':>13}{'lag1_raw':>10}{'lag1_demeaned':>15}{'p':>10}{'stable_share':>13}")
    for label, r in report.items():
        print(
            f"{label:<20}{r['n_rows']:>7}{r['var_by_turn']*100:>12.1f}%{r['lag1_raw']:>10.4f}"
            f"{r['lag1_demeaned']:>15.4f}{r['lag1_demeaned_p']:>10.2e}{r['stable_traj_share']*100:>12.1f}%"
        )


# ---------------------------------------------------------------------------
# Segment 2: activation-projection readout + input-gain OLS.
# ---------------------------------------------------------------------------


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS with classical (homoskedastic) standard errors -- no
    statsmodels in this environment, so implemented directly. `X`'s first
    column must be the intercept.
    """
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    dof = n - k
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t = beta / se
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return {"beta": beta.tolist(), "se": se.tolist(), "t": t.tolist(), "p": p.tolist(), "r2": r2, "n": n}


def build_input_gain_transitions(rows: list[dict], key_fn, x_col: str) -> tuple[np.ndarray, np.ndarray, int]:
    """x_{t+1} ~ 1 + x_t + u_remind_t + turn, over every adjacent-turn pair
    pooled across trajectories/arms.
    """
    groups = group_by_key(rows, key_fn)
    X, y_next, n_reminded = [], [], 0
    for traj in groups.values():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            X.append([1.0, float(a[x_col]), float(a["u_remind"]), float(a["turn"])])
            y_next.append(float(b[x_col]))
            if int(a["u_remind"]) != 0:
                n_reminded += 1
    return np.array(X), np.array(y_next), n_reminded


def run_segment2() -> dict:
    data = json.loads(pathlib.Path(REFUSAL_DIRECTION_READOUT).read_text())
    rows = data["rows"]
    key_fn = lambda row: row["arm"] + "|" + row["trajectory_id"]  # noqa: E731

    report = {"n_rows": len(rows), "arms": sorted({row["arm"] for row in rows}), "readouts": {}}
    for col in ("proj_pre_reply", "proj_post_reply", "y_safety"):
        decomposition = variance_decomposition(rows, y_col=col, key_fn=key_fn)
        X, y_next, n_reminded = build_input_gain_transitions(rows, key_fn, col)
        gain = ols(X, y_next)
        report["readouts"][col] = {
            **decomposition,
            "input_gain": {
                "formula": "x_{t+1} ~ 1 + x_t + u_remind_t + turn",
                "intercept": gain["beta"][0],
                "self_persistence_coef": gain["beta"][1],
                "u_remind_coef": gain["beta"][2],
                "u_remind_se": gain["se"][2],
                "u_remind_p": gain["p"][2],
                "turn_coef": gain["beta"][3],
                "r2": gain["r2"],
                "n_transitions": gain["n"],
                "n_reminded_transitions": n_reminded,
            },
        }
    return report


def print_segment2(report: dict) -> None:
    print("\n=== segment 2: activation-projection readout (320 rows, 4 Phase J arms) ===")
    print(f"{'readout':<16}{'var_by_turn':>13}{'lag1_demeaned':>15}{'p':>10}{'stable_share':>13}")
    for col, r in report["readouts"].items():
        print(
            f"{col:<16}{r['var_by_turn']*100:>12.1f}%{r['lag1_demeaned']:>15.4f}{r['lag1_demeaned_p']:>10.2e}"
            f"{r['stable_traj_share']*100:>12.1f}%"
        )
    print(f"\n{'readout':<16}{'self-persist':>13}{'u_remind coef':>15}{'p':>10}{'R2':>8}{'reminded/total':>16}")
    for col, r in report["readouts"].items():
        gain = r["input_gain"]
        print(
            f"{col:<16}{gain['self_persistence_coef']:>13.4f}{gain['u_remind_coef']:>15.4f}"
            f"{gain['u_remind_p']:>10.4f}{gain['r2']:>8.3f}"
            f"{gain['n_reminded_transitions']:>8d}/{gain['n_transitions']:<7d}"
        )


# ---------------------------------------------------------------------------
# Segment 3: the two claim gates.
# ---------------------------------------------------------------------------


def per_trajectory_arm_metrics(path: str) -> dict[str, dict]:
    rows = load_trajectories(path)
    groups = group_by_key(rows, lambda r: r["trajectory_id"])
    metrics = {}
    for tid, traj in groups.items():
        by_turn = {row["turn"]: row for row in traj}
        late = [by_turn[t]["y_safety"] for t in LATE_WINDOW if t in by_turn]
        metrics[tid] = {
            "late_y": float(np.mean(late)) if late else float("nan"),
            "terminal_y": float(by_turn[TERMINAL_TURN]["y_safety"]) if TERMINAL_TURN in by_turn else float("nan"),
            "reminders": float(sum(int(row["u_remind"]) for row in traj)),
        }
    return metrics


def paired_bootstrap(diffs: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_pairs": int(len(diffs)),
    }


def mixed_baseline_bootstrap(
    adaptive: np.ndarray,
    zero: np.ndarray,
    fixed: np.ndarray,
    spend_prob: float,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Gate 2: resample trajectory indices, then independently (per resampled
    slot) flip a spend_prob-weighted coin to decide whether that slot's
    equal-cost mixed baseline uses the fixed arm's value (spent) or
    zero_control's (unspent); diff = adaptive - mix, on the SAME resampled
    indices.
    """
    rng = np.random.default_rng(seed)
    n_traj = len(adaptive)
    idx = rng.integers(0, n_traj, size=(n_resamples, n_traj))
    treated = rng.random(size=(n_resamples, n_traj)) < spend_prob
    mix_vals = np.where(treated, fixed[idx], zero[idx])
    diffs = (adaptive[idx] - mix_vals).mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_pairs": int(n_traj),
        "spend_prob": float(spend_prob),
    }


def run_segment3() -> dict:
    self_metrics = {arm: per_trajectory_arm_metrics(paths[0]) for arm, paths in ARM_SOURCES.items()}
    indep_metrics = {arm: per_trajectory_arm_metrics(paths[1]) for arm, paths in ARM_SOURCES.items()}

    common = set.intersection(*(set(m) for m in self_metrics.values()))
    common &= set.intersection(*(set(m) for m in indep_metrics.values()))
    common = sorted(common)

    def arrays(metrics: dict[str, dict], arm: str, field: str) -> np.ndarray:
        return np.array([metrics[arm][tid][field] for tid in common])

    point_estimates = {}
    for arm in ARM_SOURCES:
        point_estimates[arm] = {
            "n_trajectories": len(common),
            "reminders_per_traj_self": float(arrays(self_metrics, arm, "reminders").mean()),
            "reminders_per_traj_indep": float(arrays(indep_metrics, arm, "reminders").mean()),
            "late_y_self": float(arrays(self_metrics, arm, "late_y").mean()),
            "late_y_indep": float(arrays(indep_metrics, arm, "late_y").mean()),
            "terminal_y_self": float(arrays(self_metrics, arm, "terminal_y").mean()),
            "terminal_y_indep": float(arrays(indep_metrics, arm, "terminal_y").mean()),
        }

    gate1 = {}
    for judge, metrics in (("self", self_metrics), ("indep", indep_metrics)):
        for field in ("late_y", "terminal_y"):
            adaptive = arrays(metrics, ADAPTIVE_ARM, field)
            zero = arrays(metrics, "zero_control", field)
            gate1[f"{judge}::{field}"] = paired_bootstrap(adaptive - zero)

    gate2 = {}
    for judge, metrics in (("self", self_metrics), ("indep", indep_metrics)):
        adaptive = arrays(metrics, ADAPTIVE_ARM, "late_y")
        zero = arrays(metrics, "zero_control", "late_y")
        spend_prob = arrays(metrics, ADAPTIVE_ARM, "reminders").mean()
        for arm in ARM_SOURCES:
            if not arm.startswith(FIXED_ARM_PREFIX):
                continue
            fixed = arrays(metrics, arm, "late_y")
            gate2[f"{judge}::{arm}"] = mixed_baseline_bootstrap(adaptive, zero, fixed, spend_prob)

    return {
        "common_trajectory_ids": common,
        "n_common_trajectories": len(common),
        "point_estimates": point_estimates,
        "gate1_adaptive_vs_zero_control": gate1,
        "gate2_adaptive_vs_equal_cost_random_mix": gate2,
    }


def print_segment3(report: dict) -> None:
    print(f"\n=== segment 3: two claim gates on {report['n_common_trajectories']} paired trajectories ===")
    if report["n_common_trajectories"] != 16:
        print("!! GATE FAILURE: expected exactly 16 common trajectories (8 held-out attacks x seeds{0,1})")

    print(f"\n{'arm':<16}{'reminders/traj (self/indep)':>30}{'late_y self':>13}{'late_y indep':>14}{'term_y self':>13}{'term_y indep':>14}")
    for arm, pe in report["point_estimates"].items():
        print(
            f"{arm:<16}{pe['reminders_per_traj_self']:>14.3f} /{pe['reminders_per_traj_indep']:>13.3f}"
            f"{pe['late_y_self']:>13.4f}{pe['late_y_indep']:>14.4f}{pe['terminal_y_self']:>13.4f}{pe['terminal_y_indep']:>14.4f}"
        )

    print(f"\ngate 1 ({ADAPTIVE_ARM} vs. zero_control):")
    for key, stats_ in report["gate1_adaptive_vs_zero_control"].items():
        passed = stats_["ci_low"] > 0 or stats_["ci_high"] < 0
        print(
            f"  {key:<20} mean_diff={stats_['mean_diff']:+.4f} 95% CI [{stats_['ci_low']:+.4f}, {stats_['ci_high']:+.4f}]"
            f" n={stats_['n_pairs']} {'PASS' if passed else 'n.s.'}"
        )

    print(f"\ngate 2 ({ADAPTIVE_ARM} vs. equal-cost random mix of zero_control + each fixed arm):")
    for key, stats_ in report["gate2_adaptive_vs_equal_cost_random_mix"].items():
        passed = stats_["ci_low"] > 0 or stats_["ci_high"] < 0
        print(
            f"  {key:<20} mean_diff={stats_['mean_diff']:+.4f} 95% CI [{stats_['ci_low']:+.4f}, {stats_['ci_high']:+.4f}]"
            f" (spend_prob={stats_['spend_prob']:.4f}) {'PASS' if passed else 'n.s.'}"
        )


# ---------------------------------------------------------------------------
# 1.3: goal-term recall (diagnostic, NOT preregistered).
# ---------------------------------------------------------------------------


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def goal_term_recall(row: dict) -> float | None:
    words = content_words(row["plain_query"])
    if not words:
        return None
    target = row["agent_message"].lower()
    hits = sum(1 for w in words if re.search(r"\b" + re.escape(w), target))
    return hits / len(words)


def run_goal_term_recall_diagnostic() -> dict:
    report = {"preregistered": False}
    for label in ("screening_self", "zero_control_self"):
        path = SEGMENT1_DATASETS[label]
        rows = load_trajectories(path)
        for row in rows:
            row["goal_term_recall"] = goal_term_recall(row)
        scored_rows = [row for row in rows if row["goal_term_recall"] is not None]
        report[label] = {"path": path, **variance_decomposition(scored_rows, y_col="goal_term_recall")}
    return report


def print_goal_term_recall(report: dict) -> None:
    print("\n=== diagnostic (preregistered=false): goal-term recall ===")
    for label, r in report.items():
        if label == "preregistered":
            continue
        print(
            f"{label:<20} lag1_demeaned={r['lag1_demeaned']:+.4f} (p={r['lag1_demeaned_p']:.2e}) "
            f"stable_traj_share={r['stable_traj_share']*100:.1f}%"
        )


# ---------------------------------------------------------------------------


def main() -> None:
    if OUT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing {OUT_PATH} -- delete it first if this is intentional")

    segment1 = run_segment1()
    segment2 = run_segment2()
    segment3 = run_segment3()
    goal_term = run_goal_term_recall_diagnostic()

    print_segment1(segment1)
    print_segment2(segment2)
    print_segment3(segment3)
    print_goal_term_recall(goal_term)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "segment1_variance_decomposition": segment1,
                "segment2_projection_readout": segment2,
                "segment3_claim_gates": segment3,
                "diagnostic_goal_term_recall": goal_term,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
