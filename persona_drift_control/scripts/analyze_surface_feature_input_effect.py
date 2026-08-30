"""CLI: Q2/Q3-style analysis (analysis.py's u_remind-effect tests) applied
to the free surface features instead of y_probe. Logic lives in
persona_drift.surface_feature_analysis; this is just argument parsing and
file I/O. Reads the trajectories_with_surface_features.jsonl already
produced by scripts/backfill_surface_features.py -- no re-extraction, no
GPU, no new generation. See that module's docstring and
docs/experiments/surface_features_backfill.md for motivation and results.

Usage:
    python scripts/analyze_surface_feature_input_effect.py \
        --augmented-path outputs/signal_screening/trajectories_with_surface_features.jsonl \
        --output-dir outputs/signal_screening
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

from persona_drift.surface_feature_analysis import analyze_input_effect, render_input_effect_markdown
from persona_drift.surface_features import SURFACE_FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--augmented-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/signal_screening/trajectories_with_surface_features.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/signal_screening"),
    )
    args = parser.parse_args()

    df = pd.read_json(args.augmented_path, lines=True)
    report = {feature: analyze_input_effect(df, feature) for feature in SURFACE_FEATURE_NAMES}

    report_path = args.output_dir / "surface_features_input_effect_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    (args.output_dir / "surface_features_input_effect_report.md").write_text(render_input_effect_markdown(report))

    print(f"wrote {report_path}")
    print(f"wrote {args.output_dir / 'surface_features_input_effect_report.md'}")


if __name__ == "__main__":
    main()
