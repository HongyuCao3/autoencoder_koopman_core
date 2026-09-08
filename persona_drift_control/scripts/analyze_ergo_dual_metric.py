#!/usr/bin/env python3
"""R2 dual-metric report (docs/experiments/ergo_fidelity_restoration_plan.md
section 4.1/4.2, section 13.3): the *only* reporting entry point for R2.
Any single-metric report of these numbers is failure mode 23.

Per arm, emits exactly three numbers (规格补丁 D1):

- `final_turn_success`: the existing legacy per-item final-turn metric,
  unchanged. REUSES `analyze_ergo_phaseC_comparison._per_item_final_turn_success`
  verbatim (via `analyze_ergo_append_comparison`, which already imports it),
  rather than reimplementing it.
- `final_attempt_success`: upstream reading (a) from 规格补丁 D1 -- the score
  at the LAST turn of the trajectory that `ergo_answer_attempt.classify_attempt`
  accepts as an answer attempt; 0.0 if no turn in the trajectory is an
  attempt (this is the "keep the denominator" reading the patch adopted,
  not "drop the trajectory": reading (b) was explicitly rejected because it
  would make each arm's n different by selecting on a quantity -- attempt
  rate -- that is itself the largest between-arm difference). Averaged
  across the 2 seeds of an item, exactly like the legacy metric.
- `attempt_rate`: fraction of the arm's TRAJECTORIES (not items -- no seed
  averaging) whose own final turn is classified as an attempt. Per the
  patch, this is not an optional diagnostic: it is what the difference
  between the other two numbers is made of.

`--gate a-b` computes the paired-by-item bootstrap (10000 resamples,
`np.random.default_rng(0)`, same recipe as the existing analyzers) on BOTH
`final_turn_success` and `final_attempt_success`, reported side by side.
Reusing `analyze_ergo_append_comparison._paired_bootstrap` and
`_assert_same_item_ids` rather than reimplementing them.

Same failure-mode-3 discipline as `analyze_ergo_append_comparison.py`: if
two arms being compared don't share the exact same item_id set, this is a
setup bug, not a silent partial comparison -- raises instead of computing
on the intersection.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from analyze_ergo_append_comparison import (  # noqa: E402
    _assert_same_item_ids,
    _lookup,
    _paired_bootstrap,
    _parse_arm_args,
    _trajectory_id_set,
)
from analyze_ergo_phaseC_comparison import _per_item_final_turn_success  # noqa: E402
from persona_drift.ergo_answer_attempt import classify_attempt  # noqa: E402
from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402


def _final_attempt_score(traj_rows: list[dict]) -> float:
    """规格补丁 D1 reading (a): the score at the trajectory's last
    attempt-classified turn, walking backward from the final turn; 0.0 if
    no turn in the trajectory is an attempt."""

    final_row = traj_rows[-1]  # group_by_trajectory sorts ascending by turn
    assert final_row["turn"] == final_row["num_shards"], "trajectory did not run to completion"
    for row in reversed(traj_rows):
        if classify_attempt(row.get("agent_message")):
            return float(row["y_task_success"])
    return 0.0


def _is_final_turn_attempt(traj_rows: list[dict]) -> bool:
    final_row = traj_rows[-1]
    return classify_attempt(final_row.get("agent_message"))


def per_item_final_attempt_success(rows: list[dict]) -> dict[str, float]:
    """Mean `_final_attempt_score` per item, averaged across the 2 seeds
    sharing an item_id -- same seed-averaging convention as
    `_per_item_final_turn_success`."""

    per_item: dict[str, list[float]] = {}
    for traj_rows in group_by_trajectory(rows).values():
        item_id = traj_rows[-1]["item_id"]
        per_item.setdefault(item_id, []).append(_final_attempt_score(traj_rows))
    return {item: float(np.mean(vals)) for item, vals in per_item.items()}


def arm_attempt_rate(rows: list[dict]) -> float:
    """Fraction of the arm's trajectories (not items) whose own final turn
    is classified as an answer attempt."""

    flags = [_is_final_turn_attempt(traj_rows) for traj_rows in group_by_trajectory(rows).values()]
    return float(np.mean(flags)) if flags else float("nan")


def summarize_arm(rows: list[dict]) -> dict[str, float]:
    per_item_legacy = _per_item_final_turn_success(rows)
    per_item_attempt = per_item_final_attempt_success(rows)
    return {
        "final_turn_success": float(np.mean(list(per_item_legacy.values()))),
        "final_attempt_success": float(np.mean(list(per_item_attempt.values()))),
        "attempt_rate": arm_attempt_rate(rows),
        "n_items": len(per_item_legacy),
        "n_trajectories": len(_trajectory_id_set(rows)),
    }


def compute_dual_gate(rows_a: list[dict], name_a: str, rows_b: list[dict], name_b: str) -> dict:
    """Paired-by-item bootstrap, `name_a - name_b`, on BOTH
    `final_turn_success` and `final_attempt_success` -- reporting only one
    is failure mode 23."""

    _assert_same_item_ids(rows_a, name_a, rows_b, name_b)

    legacy_a = _per_item_final_turn_success(rows_a)
    legacy_b = _per_item_final_turn_success(rows_b)
    common_items = sorted(set(legacy_a) & set(legacy_b))
    legacy_diffs = np.array([legacy_a[i] - legacy_b[i] for i in common_items])

    attempt_a = per_item_final_attempt_success(rows_a)
    attempt_b = per_item_final_attempt_success(rows_b)
    attempt_diffs = np.array([attempt_a[i] - attempt_b[i] for i in common_items])

    result = {
        "arm_a": name_a,
        "arm_b": name_b,
        "item_id_sets_equal": True,
        "final_turn_success": _paired_bootstrap(legacy_diffs),
        "final_attempt_success": _paired_bootstrap(attempt_diffs),
    }
    return result


def _load_arm(directory: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(directory) / "trajectories.jsonl"
    return load_trajectories(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="name=path (path is an output dir containing trajectories.jsonl), repeatable",
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        help="a-b: paired bootstrap on final_turn_success AND final_attempt_success, arm a minus arm b, repeatable",
    )
    parser.add_argument("--output", type=pathlib.Path, default=None, help="write the JSON report here (refuses to overwrite)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_dirs = _parse_arm_args(args.arm)
    loaded = {name: _load_arm(path) for name, path in arm_dirs.items()}

    report: dict[str, object] = {
        "arms": {
            name: {
                "directory": str(arm_dirs[name]),
                **summarize_arm(rows),
            }
            for name, rows in loaded.items()
        },
        "gates": {},
    }

    for spec in args.gate:
        if "-" not in spec:
            raise ValueError(f"--gate must be a-b, got {spec!r}")
        a, b = spec.split("-", 1)
        rows_a = _lookup(loaded, a, "--gate")
        rows_b = _lookup(loaded, b, "--gate")
        report["gates"][spec] = compute_dual_gate(rows_a, a, rows_b, b)

    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"\nreport written to {args.output}")


if __name__ == "__main__":
    main()
