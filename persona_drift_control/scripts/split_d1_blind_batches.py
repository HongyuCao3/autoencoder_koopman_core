#!/usr/bin/env python3
"""Split outputs/d1_blind_sample/blind_items.json into pass1/pass2 batches
of 20 (7 batches x 2 passes = 14), each pass using a fresh shuffle
(default_rng(1) / default_rng(2)) per the P protocol
(docs/experiments/measurement_validity_plan.md section P.2 / 3.1)."""

from __future__ import annotations

import json
import pathlib

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "outputs/d1_blind_sample"
BATCH_SIZE = 20


def main() -> None:
    items = json.load(open(SAMPLE_DIR / "blind_items.json"))
    n = len(items)
    for pass_idx, seed in ((1, 1), (2, 2)):
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        shuffled = [items[i] for i in order]
        batch_dir = SAMPLE_DIR / f"batches_pass{pass_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
        for b in range(n_batches):
            chunk = shuffled[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            with open(batch_dir / f"batch_{b:02d}.json", "w") as f:
                json.dump(chunk, f, indent=2)
        print(f"pass{pass_idx}: {n_batches} batches written to {batch_dir}")


if __name__ == "__main__":
    main()
