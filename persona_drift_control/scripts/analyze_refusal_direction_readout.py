#!/usr/bin/env python3
"""Readout (1) vs readout (2): does the activation projection agree with the judge?

ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7's readout list puts the
activation projection onto a calibrated refusal/safety direction FIRST ("免费
连续量"), with the LLM judge second as "外部校验", and its step 1 asks to
"顺带确认 refusal 方向投影与 judge 分的相关性(readout 有效性)". That check was
never run. This script runs it, on already-collected trajectories.

Why it matters now: y_safety takes 5 values (bin width 0.25) and 61% of Phase
J's scores sit at the 1.00 ceiling, while the differences Phase J had to
resolve were 1-2 judge bins across 16 trajectories. The projection is
deterministic (same reply -> same hidden state -> same number, no sampling and
no second model's judgement), continuous, and recomputable from the stored
`agent_message` without re-running any conversation.

DISJOINTNESS IS ENFORCED, not assumed: the direction's harmless pole is the
turn-1 queries of the attacks it was calibrated on, so a direction fit on the
attacks being scored is partly fit to call their opening turns safe. The
2026-08-31 artifact overlaps all 8 Phase J held-out attacks, so this script
refuses to use a direction whose calibration set intersects the arms' attacks
(--allow-overlap to override for a deliberate sanity check).

Two projection positions are reported because only the correlation can say
which is the right one:
  pre_reply   activation with the conversation ending at the user turn -- the
              same position the direction was calibrated at, so the most
              defensible one, but it is a PREDICTION (the reply does not exist
              yet), not a measurement of the reply.
  post_reply  activation with the model's own reply appended -- what the
              feasibility document's "每轮回复激活" literally asks for, but a
              position the direction was never calibrated at.

Needs the agent model on a GPU (one forward pass per row, no generation).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from scipy import stats  # noqa: E402

from persona_drift.chat_model import ChatModel  # noqa: E402
from persona_drift.modeling.dataset import group_by_trajectory, load_trajectories  # noqa: E402

DEFAULT_ARMS = {
    "phaseJ_koopman": "outputs/koopman_defense_phaseJ_budget1_koopman",
    "phaseJ_threshold": "outputs/koopman_defense_phaseJ_budget1_threshold",
    "phaseJ_fixed_t4": "outputs/koopman_defense_phaseJ_budget1_fixed_t4",
    "phaseJ_fixed_t1": "outputs/koopman_defense_phaseJ_budget1_fixed_t1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", default=None, metavar="NAME=DIR")
    parser.add_argument("--direction-path", type=pathlib.Path, required=True)
    parser.add_argument("--direction-stats-path", type=pathlib.Path, default=None)
    parser.add_argument("--layer", type=int, required=True, help="must match the direction's calibration layer")
    parser.add_argument("--agent-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument(
        "--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/refusal_direction_readout.json")
    )
    return parser.parse_args()


def rebuild_turn_messages(traj_rows: list[dict]) -> list[tuple[int, list[dict], list[dict]]]:
    """Reconstructs, per turn, the exact agent-facing conversation
    trajectory_runner.run_reminder_gated_trajectory built at that turn:
    `inserted_reminder_text` prefixed to the turn's stimulus on a line of its
    own, then the model's reply appended before the next turn. Returns
    (turn, history_ending_at_user_turn, history_including_reply)."""

    ordered = sorted(traj_rows, key=lambda r: r["turn"])
    history: list[dict] = []
    out = []
    for row in ordered:
        stimulus = row.get("attacker_query") or row.get("question")
        reminder = row.get("inserted_reminder_text")
        agent_facing = f"{reminder}\n{stimulus}" if reminder else stimulus
        history = history + [{"role": "user", "content": agent_facing}]
        pre = list(history)
        history = history + [{"role": "assistant", "content": row["agent_message"]}]
        out.append((int(row["turn"]), pre, list(history)))
    return out


def main() -> None:
    args = parse_args()
    arms = dict(pair.split("=", 1) for pair in args.arm) if args.arm else DEFAULT_ARMS

    direction = np.load(args.direction_path)
    stats_path = args.direction_stats_path or args.direction_path.with_name("safety_direction_stats.json")
    direction_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    calibration_ids = set(direction_stats.get("calibration_attack_ids", []))
    if direction_stats.get("layer") is not None and direction_stats["layer"] != args.layer:
        raise SystemExit(
            f"--layer {args.layer} does not match the direction's calibration layer "
            f"{direction_stats['layer']}: the projection would be onto a direction from a "
            "different residual-stream position and means nothing"
        )

    loaded: dict[str, list[dict]] = {}
    for name, directory in arms.items():
        path = pathlib.Path(directory) / "trajectories.jsonl"
        if not path.exists():
            print(f"!! {name}: {path} missing, skipped")
            continue
        loaded[name] = load_trajectories(path)

    scored_attacks = {row["attack_id"] for rows in loaded.values() for row in rows}
    overlap = sorted(scored_attacks & calibration_ids)
    if overlap and not args.allow_overlap:
        raise SystemExit(
            f"direction was calibrated on {len(overlap)} of the {len(scored_attacks)} attacks it "
            f"would now score: {overlap}\nThe harmless pole IS those attacks' turn-1 queries, so "
            "the projection would partly measure its own fit. Recalibrate with "
            "`calibrate_safety_direction.py --exclude-attack-ids " + ",".join(sorted(scored_attacks)) + "` "
            "or pass --allow-overlap if you specifically want the contaminated number."
        )
    print(f"direction: layer={direction_stats.get('layer')} dim={direction.shape[0]} "
          f"calibrated on {len(calibration_ids)} attacks, overlap with scored attacks = {len(overlap)}")

    logger.info("loading agent model {} (one forward pass per row, no generation)", args.agent_model)
    agent = ChatModel(args.agent_model, device=args.device)

    records: list[dict] = []
    for name, rows in loaded.items():
        for tid, traj in group_by_trajectory(rows).items():
            by_turn = {int(r["turn"]): r for r in traj}
            for turn, pre, post in rebuild_turn_messages(traj):
                records.append({
                    "arm": name,
                    "trajectory_id": tid,
                    "attack_id": by_turn[turn]["attack_id"],
                    "seed": by_turn[turn]["seed"],
                    "turn": turn,
                    "y_safety": float(by_turn[turn]["y_safety"]),
                    "u_remind": int(by_turn[turn]["u_remind"]),
                    "proj_pre_reply": float(np.dot(agent.hidden_state_at_layer(pre, args.layer), direction)),
                    "proj_post_reply": float(np.dot(agent.hidden_state_at_layer(post, args.layer), direction)),
                })
        logger.info("{}: {} rows projected", name, sum(1 for r in records if r["arm"] == name))

    df = pd.DataFrame(records)

    print(f"\n=== agreement with the judge (n={len(df)} rows) ===")
    for column in ("proj_pre_reply", "proj_post_reply"):
        rho, rho_p = stats.spearmanr(df[column], df["y_safety"])
        r, r_p = stats.pearsonr(df[column], df["y_safety"])
        print(f"{column:<16} Spearman rho={rho:+.3f} (p={rho_p:.2e})   Pearson r={r:+.3f} (p={r_p:.2e})")

    print("\n=== does it separate the judge's 5 levels? (mean projection per level) ===")
    print(f"{'y_safety':>9}{'n':>6}{'pre_reply':>14}{'post_reply':>14}")
    for level, g in df.groupby("y_safety"):
        print(f"{level:>9.2f}{len(g):>6}{g['proj_pre_reply'].mean():>14.2f}{g['proj_post_reply'].mean():>14.2f}")

    print("\n=== per turn (turn 1 is the judge's zero-variance ceiling turn) ===")
    print(f"{'turn':>5}{'n':>6}{'y_safety sd':>13}{'pre sd':>12}{'post sd':>12}{'rho(post)':>11}")
    for turn, g in df.groupby("turn"):
        rho = stats.spearmanr(g["proj_post_reply"], g["y_safety"])[0] if g["y_safety"].nunique() > 1 else float("nan")
        print(f"{turn:>5}{len(g):>6}{g['y_safety'].std():>13.4f}{g['proj_pre_reply'].std():>12.2f}"
              f"{g['proj_post_reply'].std():>12.2f}{rho:>11.3f}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps({
        "direction_stats": direction_stats,
        "layer": args.layer,
        "overlap_with_scored_attacks": overlap,
        "rows": records,
    }, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
