#!/usr/bin/env python3
"""Thin CLI for D1's baseline-rate screening
(docs/experiments/two_task_success_plan.md section four, D1). Unlike
scripts/run_adversarial_screening.py, this does not do its own
category-stratified num_attacks sampling -- it takes an explicit,
externally-determined attack-id list (see scripts/select_d1_attacks.py) and
passes it straight through to `run_adversarial_screening()`'s existing
`attack_ids` parameter (src/persona_drift/adversarial_screening.py:44,
already used for Phase C's held-out split). Deliberately does not
duplicate scripts/run_adversarial_screening.py's sampling logic -- that CLI
is left untouched (see the D1 handoff: modifying it was ruled out, and its
sampling path is exactly what D1 needs to bypass).

Zero-control only (no `controller_factory` override), no judge dependency
beyond whatever `run_adversarial_screening` does by default (self-judging,
--judge-model defaults to --agent-model, unused for D1's blind-labeling
protocol but left in place since the underlying function always judges).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.adversarial_screening import run_adversarial_screening  # noqa: E402
from persona_drift.attack_trajectory import AttackTrajectoryConfig  # noqa: E402
from persona_drift.chat_model import GenerationConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --agent-model (self-judging, no second model load); D1 does not use judge scores",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--attack-ids-file",
        type=pathlib.Path,
        required=True,
        help="space/newline-separated attack_id list, e.g. conf/experiment/d1_attack_ids.txt",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--agent-max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_model = args.judge_model or args.agent_model
    attack_ids = args.attack_ids_file.read_text().split()
    if not attack_ids:
        raise ValueError(f"no attack ids read from {args.attack_ids_file}")

    trajectory_config = AttackTrajectoryConfig(
        agent_gen=GenerationConfig(max_new_tokens=args.agent_max_new_tokens),
    )
    report = run_adversarial_screening(
        agent_model_id=args.agent_model,
        judge_model_id=judge_model,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        device=args.device,
        trajectory_config=trajectory_config,
        attack_ids=attack_ids,
    )
    print(f"attack_ids: {len(attack_ids)} from {args.attack_ids_file}")
    print(f"new_q1_escalation.pass={report['new_q1_escalation']['pass']}")
    print(f"new_q3_autocorrelation.pass={report['new_q3_autocorrelation']['pass']}")
    print(f"report written to {args.output_dir}/adversarial_screening_report.md")


if __name__ == "__main__":
    main()
