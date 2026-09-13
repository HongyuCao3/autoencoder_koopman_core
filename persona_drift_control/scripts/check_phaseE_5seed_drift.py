#!/usr/bin/env python3
"""Did the harness refactor or twelve days of environment drift move a byte?

G1 reran `defense`'s two endpoint arms as a whole 5 seeds into a fresh
directory instead of appending seeds 2-4 to the 2-seed Phase E products
(docs/LEDGER.md, G1 section). The stated reason was that those Phase E
directories were written by the CLI runner twelve days earlier and carry no
`hydra_run_config.json`, so `run_config_guard` has nothing to compare against
and an append would fold an unverifiable environment delta into the
cross-seed spread -- invisibly.

Rerunning all five seeds buys back the comparison this script performs:
seeds 0 and 1 exist on both sides, produced by two different code paths
(the turn loop moved into `trajectory_runner.run_reminder_gated_trajectory`,
the 1-5 judge parse into `judge_scoring.parse_1_to_5_score`) twelve days
apart, so the overlap is a direct measurement of how much that cost.

Generation is greedy and seeded, so the honest expectation is byte-identical.
That makes this a cheap check with a sharp failure mode rather than a
statistic: any mismatch at all means either the refactor was not behaviour
preserving or the environment moved, and the 5-seed products are then the
only ones safe to report from.

CPU-only. Exits non-zero when anything differs, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEFAULT_PAIRS = {
    "zero_control": (
        "outputs/koopman_defense_phaseE_zero_control",
        "outputs/koopman_defense_phaseE_zero_control_5seed",
    ),
    "constant_remind": (
        "outputs/koopman_defense_phaseE_constant_remind",
        "outputs/koopman_defense_phaseE_constant_remind_5seed",
    ),
}
# The fields a drift would show up in, in the order a reader should care:
# the prompt the agent saw, what it said, and what the judge made of it.
SCORES_NAME = "trajectories.jsonl"
COMPARED_FIELDS = ("attacker_query", "agent_message", "y_safety", "u_remind", "refusal_flag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        metavar="NAME=ARCHIVED_DIR:RERUN_DIR",
        help="repeatable; overrides the built-in pair list entirely when given",
    )
    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/phaseE_5seed_drift_check.json"),
    )
    return parser.parse_args()


def _index(path: pathlib.Path) -> dict[tuple, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {(row["attack_id"], row["seed"], row["turn"]): row for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"{path}: (attack_id, seed, turn) is not unique -- {len(rows)} rows, {len(indexed)} keys")
    return indexed


def compare(archived: pathlib.Path, rerun: pathlib.Path) -> dict:
    old, new = _index(archived / SCORES_NAME), _index(rerun / SCORES_NAME)
    shared = sorted(set(old) & set(new))
    if not shared:
        raise SystemExit(f"{archived} and {rerun} share no (attack_id, seed, turn) key -- not the same arm")
    # A rerun that dropped seeds rather than adding them would still produce a
    # clean overlap report, so say what the overlap is made of.
    mismatches = {field: [] for field in COMPARED_FIELDS}
    for key in shared:
        for field in COMPARED_FIELDS:
            if old[key][field] != new[key][field]:
                mismatches[field].append({"key": list(key), "archived": old[key][field], "rerun": new[key][field]})
    return {
        "archived": str(archived),
        "rerun": str(rerun),
        "n_archived_rows": len(old),
        "n_rerun_rows": len(new),
        "n_shared_rows": len(shared),
        "shared_seeds": sorted({key[1] for key in shared}),
        "rerun_only_seeds": sorted({key[1] for key in new} - {key[1] for key in old}),
        "decoding_config_equal": old[shared[0]]["decoding_config"] == new[shared[0]]["decoding_config"],
        "judge_model_equal": old[shared[0]]["judge_model"] == new[shared[0]]["judge_model"],
        "n_mismatched": {field: len(hits) for field, hits in mismatches.items()},
        # Two examples is enough to tell a systematic shift from a stray flake;
        # the full list would be the dataset over again.
        "examples": {field: hits[:2] for field, hits in mismatches.items() if hits},
        "identical": all(not hits for hits in mismatches.values()),
    }


def main() -> None:
    args = parse_args()
    pairs = (
        {name: tuple(spec.split(":", 1)) for name, spec in (item.split("=", 1) for item in args.pair)}
        if args.pair
        else DEFAULT_PAIRS
    )

    report = {}
    for name, (archived, rerun) in pairs.items():
        report[name] = compare(pathlib.Path(archived), pathlib.Path(rerun))
        arm = report[name]
        verdict = "IDENTICAL" if arm["identical"] else "DRIFT"
        print(
            f"{name:<18}{arm['n_shared_rows']:>4} shared rows (seeds {arm['shared_seeds']}), "
            f"+{len(arm['rerun_only_seeds'])} new seeds  -> {verdict}"
        )
        for field in COMPARED_FIELDS:
            n_bad = arm["n_mismatched"][field]
            print(f"    {field:<16}{arm['n_shared_rows'] - n_bad:>4}/{arm['n_shared_rows']} identical")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")

    drifted = [name for name, arm in report.items() if not arm["identical"]]
    if drifted:
        raise SystemExit(
            f"DRIFT in {drifted}: the refactor is not byte-equivalent or the environment moved. "
            "The 5-seed products are then the only ones safe to report from, and the archived "
            "2-seed numbers must not be quoted alongside them."
        )
    print("no drift: the refactor is byte-equivalent and the environment did not move across the gap")


if __name__ == "__main__":
    main()
