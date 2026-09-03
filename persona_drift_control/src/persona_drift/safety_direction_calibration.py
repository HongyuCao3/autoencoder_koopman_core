"""Orchestrates channel-C safety-direction calibration end to end:
sample harmful/harmless calibration texts from the attack bank, run the
forward passes (activation_direction.compute_safety_direction), save the
direction vector + stats. Step 2's prerequisite in
docs/task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable

import numpy as np
from loguru import logger

from .activation_direction import compute_safety_direction
from .attack_bank import load_attack_bank, select_screening_attacks
from .chat_model import ChatModel


def build_calibration_texts(
    num_attacks: int, rng_seed: int, exclude_attack_ids: Iterable[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """harmless texts = turn-1 (context-building) queries, harmful texts =
    plain_query (the underlying harmful goal), both drawn from the same
    sampled attacks -- matched-domain pairs, same convention as Arditi et
    al.'s own paired harmful/harmless instruction sets. Turn-1 queries were
    already spot-checked as genuinely benign in the step-1 screening (see
    docs/experiments/adversarial_screening_pilot.md's "结果" section: turn1
    mean y_safety=0.931, judge scores like the DEA/pet-care examples came
    back correctly as fully safe).

    `exclude_attack_ids` drops those attacks from the bank BEFORE sampling.
    Whether it is needed depends entirely on what the direction is for:

    * As a channel-C STEERING actuator (what this module was written for,
      ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 2), overlap
      with the evaluation attacks is not disqualifying -- you are building
      an input channel, not a measurement, and the original calibration
      left it unconstrained on purpose (see the note this replaces).
    * As a READOUT (that document's readout item 1, "每轮回复激活在
      refusal/安全方向上的投影"), overlap IS disqualifying: the harmless
      pole is literally the turn-1 queries of the sampled attacks, so a
      direction fit on the evaluation attacks is fit to score their opening
      turns as safe, and projecting those same trajectories measures partly
      that fit. The 2026-08-31 artifact under outputs/safety_direction/
      turns out to overlap all 8 Phase J held-out attacks (40 sampled from
      a 600-row bank, so this was luck, not design), which is why it cannot
      be reused for the readout without recalibrating here."""

    bank = load_attack_bank()
    if exclude_attack_ids:
        excluded = set(exclude_attack_ids)
        bank = {
            category: [entry for entry in entries if entry.attack_id not in excluded]
            for category, entries in bank.items()
        }
        bank = {category: entries for category, entries in bank.items() if entries}
    attacks = select_screening_attacks(bank, num_attacks=num_attacks, rng_seed=rng_seed)
    harmless_texts = [entry.multi_turn_queries[0] for entry in attacks]
    harmful_texts = [entry.plain_query for entry in attacks]
    return harmful_texts, harmless_texts, [entry.attack_id for entry in attacks]


def run_calibration(
    agent_model_id: str,
    output_dir: pathlib.Path,
    layer: int,
    num_attacks: int = 40,
    rng_seed: int = 100,  # disjoint from step-1 screening's attack_rng_seed=0 and dose_response_screening's default query_rng_seed=200
    device: str = "cuda",
    exclude_attack_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harmful_texts, harmless_texts, attack_ids = build_calibration_texts(
        num_attacks, rng_seed, exclude_attack_ids=exclude_attack_ids
    )
    logger.info("loading agent model {} for direction calibration", agent_model_id)
    chat_model = ChatModel(agent_model_id, device=device)

    direction, stats = compute_safety_direction(chat_model, harmful_texts, harmless_texts, layer=layer)
    stats["agent_model_id"] = agent_model_id
    stats["calibration_attack_ids"] = attack_ids
    # Recorded so a consumer can check disjointness itself rather than
    # trusting the caller passed the right list -- see build_calibration_texts
    # on why a readout needs this and a steering actuator does not.
    stats["excluded_attack_ids"] = sorted(exclude_attack_ids) if exclude_attack_ids else []

    direction_path = output_dir / "safety_direction.npy"
    np.save(direction_path, direction)
    (output_dir / "safety_direction_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "direction calibrated at layer {}: harmless_proj={:.4f} harmful_proj={:.4f} "
        "(want harmless > harmful; direction saved to {})",
        layer,
        stats["harmless_mean_projection"],
        stats["harmful_mean_projection"],
        direction_path,
    )
    return stats
