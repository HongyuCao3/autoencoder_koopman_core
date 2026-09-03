"""Orchestrates channel-C safety-direction calibration end to end:
sample harmful/harmless calibration texts from the attack bank, run the
forward passes (activation_direction.compute_safety_direction), save the
direction vector + stats. Step 2's prerequisite in
docs/task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from loguru import logger

from .activation_direction import compute_safety_direction
from .attack_bank import load_attack_bank, select_screening_attacks
from .chat_model import ChatModel


def build_calibration_texts(num_attacks: int, rng_seed: int) -> tuple[list[str], list[str], list[str]]:
    """harmless texts = turn-1 (context-building) queries, harmful texts =
    plain_query (the underlying harmful goal), both drawn from the same
    sampled attacks -- matched-domain pairs, same convention as Arditi et
    al.'s own paired harmful/harmless instruction sets. Turn-1 queries were
    already spot-checked as genuinely benign in the step-1 screening (see
    docs/experiments/adversarial_screening_pilot.md's "结果" section: turn1
    mean y_safety=0.931, judge scores like the DEA/pet-care examples came
    back correctly as fully safe).

    Known limitation: doesn't track disjointness from whichever attacks a
    later dose-response run samples with its own rng_seed -- acceptable at
    this screening scale (600-row bank, different rng_seed per phase makes
    full overlap unlikely but not impossible), flagged rather than
    engineered around."""

    bank = load_attack_bank()
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
) -> dict[str, Any]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harmful_texts, harmless_texts, attack_ids = build_calibration_texts(num_attacks, rng_seed)
    logger.info("loading agent model {} for direction calibration", agent_model_id)
    chat_model = ChatModel(agent_model_id, device=device)

    direction, stats = compute_safety_direction(chat_model, harmful_texts, harmless_texts, layer=layer)
    stats["agent_model_id"] = agent_model_id
    stats["calibration_attack_ids"] = attack_ids

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
