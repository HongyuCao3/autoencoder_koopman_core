#!/usr/bin/env python3
"""Zero-GPU audit: classify every score_fn in the (post-KNOWN_SATURATED_
PROMPT_IDS-exclusion) prompt bank with
`prompt_bank.classify_scorer_screening_safety`, so a bigger prompt sweep can
be planned around confirmed-safe entries instead of discovering saturated/
unbounded scorers after paying for real GPU trajectories.

Run on the cluster CPU (same env as run_tests.sbatch -- no torch/GPU needed):
    python scripts/audit_prompt_scorers.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.prompt_bank import classify_scorer_screening_safety, load_prompt_bank  # noqa: E402


def main() -> None:
    bank = load_prompt_bank()
    for label, entries in bank.items():
        by_reason: dict[str, list[str]] = {}
        for entry in entries:
            _, reason = classify_scorer_screening_safety(entry.score_fn)
            by_reason.setdefault(reason, []).append(entry.prompt_id)
        print(f"\n=== {label} ({len(entries)} entries) ===")
        for reason in ("ok", "binary_across_battery", "out_of_unit_range", "all_battery_evals_failed"):
            ids = by_reason.get(reason, [])
            print(f"  {reason}: {len(ids)}  {ids}")


if __name__ == "__main__":
    main()
