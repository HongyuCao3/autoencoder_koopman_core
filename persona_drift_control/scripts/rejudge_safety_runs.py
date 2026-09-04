#!/usr/bin/env python3
"""Re-score already-collected attack trajectories with an independent judge
model (GPU: one judge forward per row, no agent generation at all).

Motivation: every judge-scored number on the defense line -- adversarial
screening, dose-response, Phase A-J -- was produced with `judge_model ==
agent_model == Qwen/Qwen3-4B` (conf/adversarial_screening.yaml's
`judge_model: null` default). The sycophancy line's paired rerun
(job 15487325) measured what that costs on its own task: one-way misses
only, roughly a constant level shift, but ~4x compression of the effect
size and a flattened turn-to-turn inertia structure
(docs/experiments/sycophancy_screening_pilot.md). The defense line's
readout post-mortem (docs/experiments/koopman_defense_pilot.md section
"读出与判据的检查") diagnosed exactly the symptoms that compression
produces -- 61% of scores pinned at 1.00, arm differences of 1-2 bins,
judge SNR 1.63 -- and tried three fixes, but never this one.

Output: `<arm>/<--out-subdir>/trajectories.jsonl`, same schema as the
source plus `*_self_judge` columns, so every existing analysis script can
be pointed at it unchanged (analyze_budget_arm_comparison.py --scores-name).
The originals are never touched.

Read scripts/compare_safety_judge_runs.py next -- it is the CPU-only half
that turns these two files into the bias numbers.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.logging_setup import configure_run_logger  # noqa: E402
from persona_drift.rejudge import rejudge_dirs  # noqa: E402

# The Phase J arms (docs/experiments/budget_constrained_defense_plan.md),
# plus the two unbudgeted references analyze_budget_arm_comparison.py
# reports alongside them. Same list, same names -- kept in sync by hand
# because that script's DEFAULT_ARMS maps names to dirs for a comparison,
# while this one only needs the directories.
DEFAULT_ARM_DIRS = [
    "outputs/koopman_defense_phaseJ_budget1_koopman",
    "outputs/koopman_defense_phaseJ_budget1_threshold",
    *[f"outputs/koopman_defense_phaseJ_budget1_fixed_t{turn}" for turn in range(1, 6)],
    "outputs/koopman_defense_phaseG_periodic",
    "outputs/koopman_defense_phaseI_koopman_mpc_valigned",
    "outputs/koopman_defense_phaseE_zero_control",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm-dir",
        action="append",
        default=None,
        help="repeatable; overrides the built-in Phase J arm list entirely when given",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="independent judge weights; the default is the same checkpoint the sycophancy "
        "line's paired rerun used (same family/generation/size, different weights), so the "
        "two lines' bias numbers stay comparable",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-name", default="trajectories.jsonl")
    parser.add_argument(
        "--out-subdir",
        default="rejudge_qwen3_4b_instruct_2507",
        help="subdirectory of each arm dir the rejudged trajectories are written to",
    )
    parser.add_argument(
        "--manifest-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_case_study/rejudge_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_dirs = [pathlib.Path(d) for d in (args.arm_dir or DEFAULT_ARM_DIRS)]
    configure_run_logger(
        f"rejudge_{args.out_subdir}",
        {"judge_model": args.judge_model, "arm_dirs": [str(d) for d in arm_dirs], "device": args.device},
    )
    manifest = rejudge_dirs(
        arm_dirs,
        judge_model_id=args.judge_model,
        out_subdir=args.out_subdir,
        device=args.device,
        source_name=args.source_name,
    )
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest written to {args.manifest_path}")
    print(
        "next: python scripts/compare_safety_judge_runs.py "
        f"--rejudge-subdir {args.out_subdir}"
    )


if __name__ == "__main__":
    main()
