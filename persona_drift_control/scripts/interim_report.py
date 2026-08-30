#!/usr/bin/env python3
"""Runs analysis.analyze_screening against a currently-in-progress
trajectories.jsonl (partial, run mid-job) so status can be checked without
waiting for the full run to finish and write screening_report.md itself.
CPU-only, no GPU/torch import needed (analysis.py is pure pandas/scipy).

    python scripts/interim_report.py outputs/drift_confirmation_pilot/trajectories.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.analysis import analyze_screening  # noqa: E402


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    trajectory_ids = sorted({row["trajectory_id"] for row in rows})
    print(f"{len(rows)} rows, {len(trajectory_ids)} trajectories (any completeness) in {path}")

    report = analyze_screening(rows)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
