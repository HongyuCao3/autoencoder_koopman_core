#!/usr/bin/env python3
"""Re-runs analyze_dose_response on an existing dose_response_rows.jsonl,
optionally restricted to a subset of alpha values -- for diagnosing whether
extreme |alpha| points (where generation can degenerate into repetition,
see docs/experiments/dose_response_pilot.md's wide-alpha result) are driving
a reported effect. Pure post-hoc analysis over already-collected rows, no
GPU/model needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.analysis_dose_response import analyze_dose_response  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=None,
        help="if given, keep only rows whose alpha is in this set (default: keep all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.rows_path.read_text().splitlines() if line.strip()]
    if args.alpha_grid is not None:
        keep = set(args.alpha_grid)
        rows = [r for r in rows if r["alpha"] in keep]
    print(f"n_rows={len(rows)} alphas={sorted({r['alpha'] for r in rows})}")
    report = analyze_dose_response(rows)
    print(json.dumps(report["new_q2_dose_response"], indent=2))


if __name__ == "__main__":
    main()
