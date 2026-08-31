#!/usr/bin/env python3
"""CLI for the escalating persona-pressure confirmation pilot
(docs/experiments/drift_confirmation_pilot.md, "下一步的小范围实验建议"). Must be
run where torch/transformers are installed and a GPU (or patient CPU) is
available - see environment/setup_env.sh. Writes trajectories.jsonl,
pressure_screening_report.json, and pressure_screening_report.md under
--output-dir.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.pressure_screening import run_pressure_screening  # noqa: E402
from persona_drift.selfchat import TrajectoryConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--user-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/pressure_screening"))
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--prompt-rng-seed", type=int, default=0)
    parser.add_argument("--num-turns", type=int, default=16)
    parser.add_argument("--probe-repeats", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_config = TrajectoryConfig(
        num_turns=args.num_turns,
        probe_repeats=args.probe_repeats,
    )
    report = run_pressure_screening(
        agent_model_id=args.agent_model,
        user_model_id=args.user_model,
        output_dir=args.output_dir,
        num_prompts=args.num_prompts,
        seeds=tuple(args.seeds),
        prompt_rng_seed=args.prompt_rng_seed,
        device=args.device,
        trajectory_config=trajectory_config,
    )
    print(f"q1_baseline_no_pressure.pass={report['q1_baseline_no_pressure']['pass']}")
    print(f"q1_escalating_pressure.pass={report['q1_escalating_pressure']['pass']}")
    print(f"report written to {args.output_dir}/pressure_screening_report.md")


if __name__ == "__main__":
    main()
