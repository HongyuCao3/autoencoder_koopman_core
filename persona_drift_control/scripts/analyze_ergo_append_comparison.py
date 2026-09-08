#!/usr/bin/env python3
"""ERGO append-mode arm comparison (docs/experiments/two_task_success_plan.md
section 2 E1 item 4, section 12.2 B1): `analyze_ergo_phaseC_comparison.py`
hardcodes `ARM_DIRS` at `outputs/ergo_math_phaseC_*` and a fixed gate set
(gate1-gate6), so it cannot compute any of E2/E3/E5's gates (G-E2-1, G-E2-2,
P1/P2/P3) or run against the append-mode output directories those gates read
from. This script generalizes the same per-item metric to an arbitrary set
of CLI-declared arm directories and two kinds of pre-registered comparisons:

- `--gate a-b`: paired-by-item bootstrap (10000 resamples,
  `np.random.default_rng(0)`, same recipe as `analyze_ergo_phaseC_comparison.
  _paired_bootstrap`) on `final_turn_success`, i.e. mean `y_task_success` at
  each trajectory's own final turn (turn == num_shards), averaged across the
  seeds sharing an item_id.
- `--identity a,b`: the fraction of trajectories where arm a and arm b's
  final-turn `judge_raw_output` is byte-for-byte identical. The denominator
  is the intersection of the two arms' `trajectory_id` sets (not item_id --
  seeds are not averaged here, since this checks whether the two arms
  produced the literal same output on the same trajectory).

Per docs/experiments/two_task_success_plan.md section 2 E1 item 4: the
per-item `final_turn_success` value is NOT reimplemented here -- it reuses
`analyze_ergo_phaseC_comparison._per_item_final_turn_success` verbatim
(including that function's `turn == num_shards` completion assertion), so
this script and analyze_ergo_phaseC_comparison.py agree on that definition
by construction rather than by two independently-written copies staying in
sync.

Failure mode 3 (docs/experiments/two_task_success_plan.md section 2 E1 item
4): if the two arms being compared don't share the exact same item_id set,
this is a setup bug (e.g. wrong --item-ids, a still-running arm), not a
silent partial comparison -- both --gate and --identity raise instead of
computing on the intersection.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from analyze_ergo_phaseC_comparison import _per_item_final_turn_success  # noqa: E402
from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402

N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 0


def _item_id_set(rows: list[dict]) -> set[str]:
    return {row["item_id"] for row in rows}


def _trajectory_id_set(rows: list[dict]) -> set[str]:
    return {row["trajectory_id"] for row in rows}


def _assert_same_item_ids(rows_a: list[dict], name_a: str, rows_b: list[dict], name_b: str) -> None:
    items_a = _item_id_set(rows_a)
    items_b = _item_id_set(rows_b)
    if items_a != items_b:
        only_a = sorted(items_a - items_b)
        only_b = sorted(items_b - items_a)
        raise ValueError(
            f"item_id sets differ between arms {name_a!r} and {name_b!r} -- stop and report "
            f"(only in {name_a}: {only_a}; only in {name_b}: {only_b})"
        )


def _final_turn_field_by_trajectory(rows: list[dict], field: str) -> dict[str, object]:
    """Each trajectory's value of `field` at its own final turn (turn ==
    num_shards). Same completion assertion as
    analyze_ergo_phaseC_comparison._per_item_final_turn_success, but keyed
    by trajectory_id (not averaged across seeds) -- identity checks whether
    two arms produced the literal same output on the same trajectory, which
    per-item seed-averaging would obscure."""

    result: dict[str, object] = {}
    for trajectory_id, traj_rows in group_by_trajectory(rows).items():
        final_row = max(traj_rows, key=lambda r: r["turn"])
        assert final_row["turn"] == final_row["num_shards"], "trajectory did not run to completion"
        result[trajectory_id] = final_row[field]
    return result


def _paired_bootstrap(diffs: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_pairs": int(len(diffs)),
        "excludes_zero_and_positive": bool(np.percentile(means, 2.5) > 0),
    }


def compute_gate(rows_a: list[dict], name_a: str, rows_b: list[dict], name_b: str) -> dict:
    """Paired-by-item bootstrap on final_turn_success, `name_a - name_b`."""

    _assert_same_item_ids(rows_a, name_a, rows_b, name_b)
    per_item_a = _per_item_final_turn_success(rows_a)
    per_item_b = _per_item_final_turn_success(rows_b)
    common_items = sorted(set(per_item_a) & set(per_item_b))
    diffs = np.array([per_item_a[i] - per_item_b[i] for i in common_items])
    result = _paired_bootstrap(diffs)
    result["arm_a"] = name_a
    result["arm_b"] = name_b
    result["item_id_sets_equal"] = True
    result["trajectory_id_intersection"] = len(_trajectory_id_set(rows_a) & _trajectory_id_set(rows_b))
    return result


def compute_identity(
    rows_a: list[dict], name_a: str, rows_b: list[dict], name_b: str, field: str = "judge_raw_output"
) -> dict:
    """Fraction of trajectories where `field` at the final turn is
    byte-for-byte identical between arm a and arm b. Denominator is the
    intersection of the two arms' trajectory_id sets."""

    _assert_same_item_ids(rows_a, name_a, rows_b, name_b)
    field_a = _final_turn_field_by_trajectory(rows_a, field)
    field_b = _final_turn_field_by_trajectory(rows_b, field)
    common_trajectories = sorted(set(field_a) & set(field_b))
    if not common_trajectories:
        raise ValueError(f"no overlapping trajectory_id between arms {name_a!r} and {name_b!r}")
    n_identical = sum(1 for tid in common_trajectories if field_a[tid] == field_b[tid])
    return {
        "arm_a": name_a,
        "arm_b": name_b,
        "field": field,
        "n_identical": n_identical,
        "n_trajectories": len(common_trajectories),
        "rate": n_identical / len(common_trajectories),
        "item_id_sets_equal": True,
        "trajectory_id_intersection": len(common_trajectories),
    }


def _load_arm(directory: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(directory) / "trajectories.jsonl"
    return load_trajectories(path)


def _parse_arm_args(specs: list[str]) -> dict[str, str]:
    arms: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--arm must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        arms[name] = path
    return arms


def _lookup(loaded: dict[str, list[dict]], name: str, flag: str) -> list[dict]:
    if name not in loaded:
        raise ValueError(f"{flag}: unknown arm {name!r}, declared arms are {sorted(loaded)}")
    return loaded[name]


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
        help="a-b: paired bootstrap on final_turn_success, arm a minus arm b, repeatable",
    )
    parser.add_argument(
        "--identity",
        action="append",
        default=[],
        help="a,b: fraction of trajectories with byte-identical final-turn judge_raw_output, repeatable",
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
                "n_rows": len(rows),
                "n_items": len(_item_id_set(rows)),
                "n_trajectories": len(_trajectory_id_set(rows)),
            }
            for name, rows in loaded.items()
        },
        "gates": {},
        "identities": {},
    }

    for spec in args.gate:
        if "-" not in spec:
            raise ValueError(f"--gate must be a-b, got {spec!r}")
        a, b = spec.split("-", 1)
        rows_a = _lookup(loaded, a, "--gate")
        rows_b = _lookup(loaded, b, "--gate")
        report["gates"][spec] = compute_gate(rows_a, a, rows_b, b)

    for spec in args.identity:
        if "," not in spec:
            raise ValueError(f"--identity must be a,b, got {spec!r}")
        a, b = spec.split(",", 1)
        rows_a = _lookup(loaded, a, "--identity")
        rows_b = _lookup(loaded, b, "--identity")
        report["identities"][spec] = compute_identity(rows_a, a, rows_b, b)

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
