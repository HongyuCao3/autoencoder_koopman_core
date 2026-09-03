#!/usr/bin/env python3
"""Cross-arm comparison for the budget-constrained setting (Phase J,
docs/experiments/budget_constrained_defense_plan.md).

Reads each arm's trajectories.jsonl and reports, per arm: reminders actually
spent, terminal-turn and late-window mean safety, and the new-Q1 erosion test
every earlier phase was judged on. Then it does the comparison the budget
setting exists to make possible -- adaptive arm vs the BEST fixed allocation,
paired per (attack, seed) trajectory with a bootstrap CI on the mean
difference.

Why the effect-size comparison and not just new-Q1: new-Q1 is a one-shot
significance test on the full-sequence OLS slope, so it is dominated by the
natural turn-1..3 decline that happens before any reactive policy can act
(Phase I's post-mortem, docs/experiments/koopman_case_study_design.md) and it
answers pass/fail rather than how much. Both are reported here -- new-Q1 for
continuity with Phase E-I, the paired effect size as the primary number --
and the seed count is still only 2, so the CIs are wide by construction (see
docs/next_step_diagnosis.md section 4 step 3, not yet executed).

CPU-only -- no GPU needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.analysis_adversarial import analyze_adversarial_screening  # noqa: E402
from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402

DEFAULT_ARMS = {
    # Phase J arms (this experiment).
    "koopman_budget1": "outputs/koopman_defense_phaseJ_budget1_koopman",
    "threshold_budget1": "outputs/koopman_defense_phaseJ_budget1_threshold",
    **{f"fixed_t{turn}_budget1": f"outputs/koopman_defense_phaseJ_budget1_fixed_t{turn}" for turn in range(1, 6)},
    # Unbudgeted references already collected, for context only -- they are NOT
    # budget-matched (periodic spends 2/trajectory, Phase I up to 2), so they
    # answer "what did the unconstrained setting look like", not "who allocates
    # a fixed budget best".
    "periodic_phaseG_unbudgeted": "outputs/koopman_defense_phaseG_periodic",
    "koopman_phaseI_unbudgeted": "outputs/koopman_defense_phaseI_koopman_mpc_valigned",
    "zero_control_phaseE": "outputs/koopman_defense_phaseE_zero_control",
}
LATE_WINDOW = (3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="repeatable; overrides the built-in arm list entirely when given",
    )
    parser.add_argument("--adaptive-arm", default="koopman_budget1")
    parser.add_argument("--fixed-arm-prefix", default="fixed_t")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/budget_arm_comparison.json")
    )
    return parser.parse_args()


def _per_trajectory_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    metrics = {}
    for tid, traj in group_by_trajectory(rows).items():
        traj = sorted(traj, key=lambda r: r["turn"])
        ys = {row["turn"]: float(row["y_safety"]) for row in traj}
        late = [ys[t] for t in LATE_WINDOW if t in ys]
        metrics[tid] = {
            "terminal_y": ys[max(ys)],
            "late_mean_y": float(np.mean(late)) if late else float("nan"),
            "mean_y": float(np.mean(list(ys.values()))),
            "n_reminders": float(sum(int(row["u_remind"]) for row in traj)),
        }
    return metrics


def _paired_bootstrap(diffs: np.ndarray, n_resamples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_pairs": int(len(diffs)),
    }


def main() -> None:
    args = parse_args()
    arms = (
        dict(pair.split("=", 1) for pair in args.arm)
        if args.arm
        else DEFAULT_ARMS
    )

    loaded: dict[str, dict] = {}
    for name, directory in arms.items():
        path = pathlib.Path(directory) / "trajectories.jsonl"
        if not path.exists():
            print(f"!! {name}: {path} missing (arm not run yet), skipped")
            continue
        rows = load_trajectories(path)
        per_traj = _per_trajectory_metrics(rows)
        report = analyze_adversarial_screening(rows)
        loaded[name] = {
            "path": str(path),
            "controller_recorded": sorted({row.get("excitation_design", "?") for row in rows}),
            "n_trajectories": len(per_traj),
            "n_reminders_total": sum(m["n_reminders"] for m in per_traj.values()),
            "terminal_y_mean": float(np.mean([m["terminal_y"] for m in per_traj.values()])),
            "late_mean_y": float(np.mean([m["late_mean_y"] for m in per_traj.values()])),
            "mean_y": float(np.mean([m["mean_y"] for m in per_traj.values()])),
            "new_q1_pass": report["new_q1_escalation"]["pass"],
            "new_q1_p": report["new_q1_escalation"]["t_test_mean_slope_vs_zero"]["p"],
            "per_trajectory": per_traj,
        }

    if not loaded:
        raise SystemExit("no arms found -- submit the Phase J sbatch scripts first (environment/run_koopman_defense_phaseJ_*)")

    print(f"{'arm':<32}{'n_traj':>7}{'reminders':>11}{'terminal_y':>12}{'late_y(3-5)':>13}{'mean_y':>9}{'new-Q1 p':>10}  q1_pass")
    for name, arm in loaded.items():
        print(
            f"{name:<32}{arm['n_trajectories']:>7}{arm['n_reminders_total']:>11.0f}"
            f"{arm['terminal_y_mean']:>12.4f}{arm['late_mean_y']:>13.4f}{arm['mean_y']:>9.4f}"
            f"{arm['new_q1_p']:>10.4f}  {arm['new_q1_pass']}"
        )

    comparisons: dict[str, dict] = {}
    adaptive = loaded.get(args.adaptive_arm)
    fixed_arms = {name: arm for name, arm in loaded.items() if name.startswith(args.fixed_arm_prefix)}
    if adaptive and fixed_arms:
        # "Best fixed allocation" is picked on the same data it is then
        # compared against, which biases the comparison AGAINST the adaptive
        # arm -- deliberately, since the claim being tested is that adaptivity
        # beats the best fixed schedule, and a baseline chosen in advance
        # could always be blamed for a bad choice of turn.
        best_fixed_name = max(fixed_arms, key=lambda name: fixed_arms[name]["late_mean_y"])
        print(f"\nbest fixed allocation by late_mean_y: {best_fixed_name}")
        for metric in ("terminal_y", "late_mean_y", "mean_y", "n_reminders"):
            for name, arm in fixed_arms.items():
                shared = sorted(set(adaptive["per_trajectory"]) & set(arm["per_trajectory"]))
                if not shared:
                    continue
                diffs = np.array(
                    [adaptive["per_trajectory"][tid][metric] - arm["per_trajectory"][tid][metric] for tid in shared]
                )
                stats = _paired_bootstrap(diffs, args.bootstrap, args.bootstrap_seed)
                comparisons[f"{args.adaptive_arm}_vs_{name}::{metric}"] = stats
                if name == best_fixed_name:
                    print(
                        f"  {metric:<13} vs {name}: mean_diff={stats['mean_diff']:+.4f} "
                        f"95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] (n={stats['n_pairs']} paired trajectories)"
                    )
        comparisons["best_fixed_arm"] = best_fixed_name

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps({"arms": loaded, "comparisons": comparisons}, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
