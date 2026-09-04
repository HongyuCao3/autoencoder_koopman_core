#!/usr/bin/env python3
"""Backfill the continuous (label-token-distribution) judge readout onto the
two already-collected sycophancy screening runs, without regenerating any
agent text (docs/experiments/continuous_readout_plan.md S3).

Reads each source directory's `trajectories.jsonl`, replays the *same*
sycophancy judge prompt through a single forward pass per row (the judge
weights are read from each row's own `judge_model`, not passed on the
command line -- see continuous_readout.score_dirs's docstring for why), and
writes `<source-dir>/<--out-name>` with the new fields alongside the
existing ones. GPU, but no generation: ~400 rows total across both default
source directories, each one 10-token judge call replaced by one forward
pass.

Read scripts/analyze_continuous_readout.py next -- it is the CPU-only half
that turns the two scored files into the G0-G3 gate report.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoTokenizer  # noqa: E402

from persona_drift.logging_setup import configure_run_logger  # noqa: E402
from persona_drift.continuous_readout import score_dirs  # noqa: E402
from persona_drift.sycophancy_judge import STANCE_LABELS, resolve_label_token_ids  # noqa: E402

DEFAULT_SOURCE_DIRS = [
    "outputs/sycophancy_screening",
    "outputs/sycophancy_screening_independent_judge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-dir",
        action="append",
        default=None,
        help="repeatable; overrides the two default sycophancy screening directories entirely when given",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-name", default="continuous_readout/trajectories.jsonl")
    parser.add_argument(
        "--manifest-path", type=pathlib.Path, default=pathlib.Path("outputs/sycophancy_continuous_readout/manifest.json")
    )
    parser.add_argument(
        "--print-label-tokens",
        metavar="MODEL_ID",
        default=None,
        help="G0 only: load just the tokenizer for MODEL_ID, print resolve_label_token_ids's "
        "result plus each candidate encoding, then exit. CPU, seconds, no GPU needed -- run this "
        "for both judge checkpoints before scoring anything.",
    )
    return parser.parse_args()


def print_label_tokens(model_id: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"model: {model_id}")
    for label in STANCE_LABELS:
        for text in (label, " " + label):
            print(f"  encode({text!r}) = {tokenizer.encode(text, add_special_tokens=False)}")
    try:
        resolved = resolve_label_token_ids(tokenizer, STANCE_LABELS)
    except ValueError as exc:
        print(f"G0 FAILED: {exc}")
        return
    print("resolved (G0 passes):")
    for label, ids in resolved.items():
        print(f"  {label}: {ids}")


def main() -> None:
    args = parse_args()
    if args.print_label_tokens:
        print_label_tokens(args.print_label_tokens)
        return

    source_dirs = [pathlib.Path(d) for d in (args.source_dir or DEFAULT_SOURCE_DIRS)]
    configure_run_logger(
        "sycophancy_continuous_readout",
        {"source_dirs": [str(d) for d in source_dirs], "device": args.device, "out_name": args.out_name},
    )
    manifest = score_dirs(source_dirs, device=args.device, out_name=args.out_name)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest written to {args.manifest_path}")
    print("next: python scripts/analyze_continuous_readout.py")


if __name__ == "__main__":
    main()
