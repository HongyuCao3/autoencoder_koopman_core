"""Diff-in-means safety/refusal direction calibration (Arditi et al.,
"Refusal in Language Models Is Mediated by a Single Direction", arXiv
2406.11717), for the adversarial-defense screening task's channel C
(docs/task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md section 7 step 2). This is the
"顺带确认 refusal 方向投影" component step 1 deliberately deferred -- see
docs/experiments/adversarial_screening_pilot.md's "有意跳过的部分" -- now
needed because step 2's steering channel requires it directly, not just as
an incidental readout check.

Direction points from the harmful pole toward the harmless/safe pole
(harmless_mean - harmful_mean), matching DATA_COLLECTION_PROTOCOL.md section
3's channel-C convention ("第 ℓ 层残差流加 α·v"): alpha>0 pushes generation
toward the observed "safe" activation region (the intended defensive
setting), alpha<0 pushes toward the observed "harmful" region (used to
sanity-check the direction's sign and to probe over-steering, not as a
defensive setting itself).

Pure numpy -- doesn't import torch itself, only calls the given chat_model's
`hidden_state_at_layer`, so the diff-in-means math is testable with a fake
activation provider (tests/test_activation_direction.py) without a real
model or GPU.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ActivationProvider(Protocol):
    def hidden_state_at_layer(self, messages: list[dict[str, str]], layer: int) -> np.ndarray: ...


def compute_safety_direction(
    chat_model: ActivationProvider,
    harmful_texts: list[str],
    harmless_texts: list[str],
    layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(harmful_texts) < 2 or len(harmless_texts) < 2:
        raise ValueError("need at least 2 texts per pole for a stable mean")

    harmful_acts = np.stack(
        [chat_model.hidden_state_at_layer([{"role": "user", "content": t}], layer) for t in harmful_texts]
    )
    harmless_acts = np.stack(
        [chat_model.hidden_state_at_layer([{"role": "user", "content": t}], layer) for t in harmless_texts]
    )
    harmful_mean = harmful_acts.mean(axis=0)
    harmless_mean = harmless_acts.mean(axis=0)
    direction = harmless_mean - harmful_mean

    stats: dict[str, Any] = {
        "layer": layer,
        "n_harmful": len(harmful_texts),
        "n_harmless": len(harmless_texts),
        "harmful_mean_norm": float(np.linalg.norm(harmful_mean)),
        "harmless_mean_norm": float(np.linalg.norm(harmless_mean)),
        "direction_norm": float(np.linalg.norm(direction)),
        # Sanity check: each pole's own mean activation, projected onto the
        # direction, should separate (harmless higher than harmful) -- if
        # this comes out backwards or near-zero, the direction isn't usable
        # for steering and step 2 should stop here rather than run a
        # dose-response sweep that will just look like noise.
        "harmful_mean_projection": float(np.dot(harmful_mean, direction)),
        "harmless_mean_projection": float(np.dot(harmless_mean, direction)),
    }
    return direction, stats
