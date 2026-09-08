#!/usr/bin/env python3
"""Selects the 100-attack pool for D1's baseline-rate screening
(docs/experiments/two_task_success_plan.md section four, D1) per Opus's
ruling (two_task_success_plan.md handoff, D1 exclusion/stratification
decision):

1. Exclusion set = union of
   (a) every attack_id that appears in any outputs/*/trajectories.jsonl
       (i.e. was actually run through the adversarial-screening pipeline
       somewhere in the project), and
   (b) the `calibration_attack_ids` recorded in both
       outputs/safety_direction/safety_direction_stats.json and
       outputs/safety_direction_readout_heldout_excluded/safety_direction_stats.json
       (attacks used to *fit* a steering direction, even if never run as a
       zero-control trajectory).
   This must come out to 79 (30 overlap between (a)'s 49 and (b)'s 60) --
   if it doesn't, stop rather than silently proceeding (Opus's ruling).

2. From the remaining pool (600 - 79 = 521), draw 100 attacks stratified
   proportionally by the upstream dataset's `category` field, using a
   single numpy.random.default_rng(0) stream consumed category-by-category
   in sorted category-name order. Per-category counts are apportioned by
   largest-remainder (Hamilton) rounding so they sum to exactly 100.

Writes:
  - conf/experiment/d1_attack_ids.txt   (100 ids, space-separated, one line
    -- same convention as conf/experiment/ergo_phaseC_item_ids.txt so a
    sbatch script can do `$(cat ...)`)
  - stdout: exclusion-set size check, per-category allocation table, and
    the category distribution of the new 100 vs. the empirical 49 (used
    directly in the D1 handoff report).
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.attack_bank import load_attack_bank  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def used_attack_ids_from_trajectories() -> set[str]:
    ids: set[str] = set()
    for path_str in sorted(glob.glob(str(REPO_ROOT / "outputs" / "*" / "trajectories.jsonl"))):
        path = pathlib.Path(path_str)
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                attack_id = row.get("attack_id")
                if attack_id:
                    ids.add(attack_id)
    return ids


def calibration_attack_ids() -> set[str]:
    ids: set[str] = set()
    for rel in (
        "outputs/safety_direction/safety_direction_stats.json",
        "outputs/safety_direction_readout_heldout_excluded/safety_direction_stats.json",
    ):
        path = REPO_ROOT / rel
        stats = json.loads(path.read_text())
        ids.update(stats["calibration_attack_ids"])
    return ids


def apportion_largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Hamilton/largest-remainder apportionment of `total` seats across
    categories weighted by `counts` (pool size per category)."""

    grand = sum(counts.values())
    raw = {cat: total * n / grand for cat, n in counts.items()}
    floor = {cat: int(np.floor(v)) for cat, v in raw.items()}
    remainder = total - sum(floor.values())
    # break ties deterministically by (fractional part desc, category name asc)
    order = sorted(raw.keys(), key=lambda c: (-(raw[c] - floor[c]), c))
    alloc = dict(floor)
    for cat in order[:remainder]:
        alloc[cat] += 1
    return alloc


def category_distribution(ids: set[str] | list[str], entry_by_id: dict[str, object]) -> Counter:
    return Counter(entry_by_id[i].category for i in ids if i in entry_by_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "conf" / "experiment" / "d1_attack_ids.txt",
    )
    parser.add_argument("--num-attacks", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    bank = load_attack_bank()
    entry_by_id = {entry.attack_id: entry for entries in bank.values() for entry in entries}
    all_ids = set(entry_by_id.keys())
    print(f"attack bank total: {len(all_ids)} attacks across {len(bank)} categories")

    used = used_attack_ids_from_trajectories()
    calib = calibration_attack_ids()
    overlap = used & calib
    exclusion = used | calib

    print(f"used (appear in some outputs/*/trajectories.jsonl): {len(used)}")
    print(f"calibration_attack_ids (both safety_direction_stats.json): {len(calib)}")
    print(f"used & calib overlap: {len(overlap)}")
    print(f"exclusion union: {len(exclusion)}")

    if len(exclusion) != 79:
        print(
            f"STOP: exclusion union size is {len(exclusion)}, expected 79 per Opus's ruling. "
            "Not proceeding to sample -- report this number and stop.",
            file=sys.stderr,
        )
        sys.exit(1)

    remaining = all_ids - exclusion
    print(f"remaining pool after exclusion: {len(remaining)}")
    if len(remaining) != 521:
        print(
            f"NOTE: remaining pool is {len(remaining)}, expected 521 (600 - 79). "
            "Proceeding since the exclusion-union check above passed, but flag this in the report.",
            file=sys.stderr,
        )

    # Per-category pool sizes within the remaining set.
    remaining_by_cat: dict[str, list[str]] = {}
    for attack_id in remaining:
        cat = entry_by_id[attack_id].category
        remaining_by_cat.setdefault(cat, []).append(attack_id)
    pool_counts = {cat: len(ids) for cat, ids in remaining_by_cat.items()}

    alloc = apportion_largest_remainder(pool_counts, args.num_attacks)
    print("\nper-category allocation for the new 100 (Hamilton apportionment, remainder ties broken by category name):")
    for cat in sorted(pool_counts):
        print(f"  {cat}: pool={pool_counts[cat]:>4} -> draw={alloc[cat]:>3}")
    assert sum(alloc.values()) == args.num_attacks

    rng = np.random.default_rng(args.rng_seed)
    selected: list[str] = []
    for cat in sorted(pool_counts):  # fixed, deterministic category order
        ids_sorted = sorted(remaining_by_cat[cat])  # fixed order before shuffling draw
        k = alloc[cat]
        if k > 0:
            idx = rng.choice(len(ids_sorted), size=k, replace=False)
            selected.extend(ids_sorted[i] for i in idx)

    assert len(selected) == args.num_attacks
    assert len(set(selected)) == args.num_attacks
    assert not (set(selected) & exclusion)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(" ".join(selected) + "\n")
    print(f"\nwrote {len(selected)} attack ids to {args.out}")
    print(f"first 5: {selected[:5]}")

    print("\ncategory distribution, new 100:")
    new_dist = category_distribution(selected, entry_by_id)
    for cat in sorted(new_dist):
        print(f"  {cat}: {new_dist[cat]}")

    print("\ncategory distribution, old 49 (empirical union of attack_ids used in outputs/*/trajectories.jsonl):")
    old_dist = category_distribution(used, entry_by_id)
    for cat in sorted(old_dist):
        print(f"  {cat}: {old_dist[cat]}")


if __name__ == "__main__":
    main()
