#!/usr/bin/env python3
"""CLI for channel-C safety-direction calibration
(ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 2 prerequisite).
Must be run where torch/transformers are installed and a GPU (or patient
CPU) is available - see environment/setup_env.sh. Writes
safety_direction.npy and safety_direction_stats.json under --output-dir;
run_dose_response_screening.py's --direction-path points at the former.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.safety_direction_calibration import run_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/safety_direction"))
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="decoder layer index (1-indexed into output_hidden_states, i.e. output of "
        "decoder layer L); pick something mid-stack (e.g. ~half of num_hidden_layers) as a "
        "starting point per Arditi et al.",
    )
    parser.add_argument("--num-attacks", type=int, default=40)
    parser.add_argument("--rng-seed", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_calibration(
        agent_model_id=args.agent_model,
        output_dir=args.output_dir,
        layer=args.layer,
        num_attacks=args.num_attacks,
        rng_seed=args.rng_seed,
        device=args.device,
    )
    print(f"harmless_mean_projection={stats['harmless_mean_projection']:.4f}")
    print(f"harmful_mean_projection={stats['harmful_mean_projection']:.4f}")
    print(f"direction_norm={stats['direction_norm']:.4f}")
    print(f"direction saved under {args.output_dir}/safety_direction.npy")


if __name__ == "__main__":
    main()
