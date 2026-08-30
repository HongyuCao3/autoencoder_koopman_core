"""CLI: backfill free surface features onto an already-completed
trajectories.jsonl and run the zero_control drift test on each. Logic lives
in persona_drift.surface_feature_analysis (importable/testable); this is
just argument parsing and file I/O. See that module's docstring and
docs/experiments/surface_features_backfill.md for motivation and results.

Usage:
    python scripts/backfill_surface_features.py \
        --trajectories-path outputs/signal_screening/trajectories.jsonl \
        --output-dir outputs/signal_screening
"""

from __future__ import annotations

import argparse
import json
import pathlib

from persona_drift.surface_feature_analysis import (
    analyze_zero_control_drift,
    backfill_dataframe,
    render_drift_markdown,
)
from persona_drift.surface_features import SURFACE_FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectories-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/signal_screening/trajectories.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/signal_screening"),
    )
    args = parser.parse_args()

    df = backfill_dataframe(args.trajectories_path)

    augmented_path = args.output_dir / "trajectories_with_surface_features.jsonl"
    df.to_json(augmented_path, orient="records", lines=True)

    report = {feature: analyze_zero_control_drift(df, feature) for feature in SURFACE_FEATURE_NAMES}
    report_path = args.output_dir / "surface_features_drift_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (args.output_dir / "surface_features_drift_report.md").write_text(render_drift_markdown(report))

    print(f"wrote {augmented_path}")
    print(f"wrote {report_path}")
    print(f"wrote {args.output_dir / 'surface_features_drift_report.md'}")


if __name__ == "__main__":
    main()
