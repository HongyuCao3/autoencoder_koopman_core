"""Offline re-scoring of already-collected sycophancy trajectories with the
continuous (label-token-distribution) judge readout instead of the greedy
hard label (docs/experiments/continuous_readout_plan.md).

Same shape as rejudge.py's offline-rescoring pattern -- every row already
stores the four strings the judge prompt is built from (`question` /
`correction` / `presupposition` / `agent_message`), so the judge prompt can
be replayed against the *same* weights that produced the stored
`stance_label`/`judge_raw_output` without regenerating any agent text. The
three domain-agnostic pieces of that pattern (JSONL loading tolerant of a
torn final line, the (trajectory_id, turn) row key, and "which rows still
need work given what a previous run already wrote") are identical to
rejudge.py's -- they don't know or care what a judge score's shape is -- so
they're imported from there rather than copied; only `score_row` differs,
because it computes a distribution over labels instead of re-running a
different judge's greedy score.

Unlike rejudge.py, this module scores the SAME judge that produced the
stored hard label (the continuous readout is a re-interpretation of the
existing judge call, not a comparison against a different judge), so
score_row asserts `row["judge_model"] == judge.model_id` and refuses to
proceed on a mismatch -- see its docstring for why that's the single easiest
mistake to make here.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Callable

import numpy as np
from loguru import logger

from .chat_model import ChatModel
from .rejudge import load_jsonl, pending_rows, row_key
from .sycophancy_judge import STANCE_LABELS, judge_sycophancy_distribution, resolve_label_token_ids

CONTINUOUS_READOUT_VERSION = "v0.1"


def score_row(judge: Any, row: dict[str, Any], label_token_ids: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """One row, scored by the label-token distribution. Returns a new dict;
    `row` is not mutated, and no existing field is overwritten -- the new
    fields all use new names (`y_consistency_continuous` etc.), so the
    original hard-label columns (`y_consistency` / `stance_label` /
    `is_flip` / `judge_raw_output`) survive untouched for downstream code
    that still reads them.

    The judge_model assertion below is the easiest mistake to make in this
    module and the hardest to notice after the fact: the two stored files
    were hard-labeled by two *different* judge checkpoints
    (Qwen/Qwen3-4B for the self-judge run, Qwen/Qwen3-4B-Instruct-2507 for
    the independent-judge run). Scoring a row with the wrong model would
    still produce a plausible-looking distribution -- there is no crash to
    catch it -- it would just be comparing two different models' opinions
    under the label `stance_label_argmax` vs `stance_label`, silently
    invalidating the G1 fidelity check this whole plan rests on."""

    if row["judge_model"] != judge.model_id:
        raise ValueError(
            f"row's judge_model {row['judge_model']!r} does not match the loaded judge "
            f"{judge.model_id!r} -- the continuous readout must be scored by the same "
            "weights that produced the row's hard label"
        )

    y_continuous, probs, total, argmax_label = judge_sycophancy_distribution(
        judge, row["question"], row["correction"], row["presupposition"], row["agent_message"], label_token_ids
    )
    scored = dict(row)
    scored["y_consistency_continuous"] = y_continuous
    scored["p_maintains"] = probs["MAINTAINS"]
    scored["p_hedges"] = probs["HEDGES"]
    scored["p_flips"] = probs["FLIPS"]
    scored["label_mass_total"] = total
    scored["stance_label_argmax"] = argmax_label
    scored["continuous_readout_version"] = CONTINUOUS_READOUT_VERSION
    return scored


def score_file(
    judge: Any,
    source_path: pathlib.Path,
    dest_path: pathlib.Path,
    label_token_ids: dict[str, tuple[int, ...]],
    log_every: int = 25,
) -> list[dict[str, Any]]:
    """Scores every row of `source_path` into `dest_path`, appending as it
    goes (a killed job keeps everything already written; resubmitting picks
    up the rest). A row counts as already done only if a previous run's
    output for that (trajectory_id, turn) both carries the same
    `agent_message` *and* already has `y_consistency_continuous` --
    filtering to the latter before handing the rows to
    rejudge.pending_rows is what makes "not yet scored" and "source
    rewritten with different text" both count as pending, exactly the
    distinction rejudge.pending_rows's docstring warns must not be
    collapsed."""

    source_rows = load_jsonl(source_path)
    source_keys = {row_key(r) for r in source_rows}
    already_scored = [
        row for row in load_jsonl(dest_path) if row_key(row) in source_keys and "y_consistency_continuous" in row
    ]
    todo = pending_rows(source_rows, already_scored)
    todo_keys = {row_key(r) for r in todo}
    kept = [row for row in already_scored if row_key(row) not in todo_keys]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    logger.info("{}: {} rows total, {} already scored, {} to do", source_path, len(source_rows), len(kept), len(todo))
    start = time.monotonic()
    with dest_path.open("a") as handle:
        for i, row in enumerate(todo, start=1):
            scored = score_row(judge, row, label_token_ids)
            handle.write(json.dumps(scored) + "\n")
            kept.append(scored)
            if i % log_every == 0 or i == len(todo):
                handle.flush()
                logger.info("  [{}/{}] {} (+{:.0f}s)", i, len(todo), row_key(row), time.monotonic() - start)
    return kept


def score_dirs(
    source_dirs: list[pathlib.Path],
    device: str = "cuda",
    out_name: str = "continuous_readout/trajectories.jsonl",
    chat_model_cls: Callable[..., Any] = ChatModel,
) -> dict[str, dict[str, Any]]:
    """Scores each source directory's `trajectories.jsonl` into
    `<dir>/<out_name>`. Judge weights are read from each directory's own
    file (`judge_model` on its first row), not taken as a command-line
    argument: the two runs this is meant for were hard-labeled by two
    different checkpoints, and score_row's assertion only protects a run
    that is fed the right judge for the right directory in the first place
    -- accepting the model id as a parameter would just move the
    opportunity to pass the wrong one from here to the caller. Directories
    are grouped by judge_model so each checkpoint is loaded at most once."""

    present = [d for d in source_dirs if (d / "trajectories.jsonl").exists()]
    missing = [str(d) for d in source_dirs if d not in present]
    for path in missing:
        logger.warning("{}/trajectories.jsonl missing, skipped", path)
    if not present:
        raise SystemExit("no source directory has a trajectories.jsonl to score")

    dir_judge_model = {d: load_jsonl(d / "trajectories.jsonl")[0]["judge_model"] for d in present}

    manifest: dict[str, dict[str, Any]] = {}
    for judge_model_id in sorted(set(dir_judge_model.values())):
        dirs_for_model = [d for d in present if dir_judge_model[d] == judge_model_id]
        logger.info("loading judge model {} on {}", judge_model_id, device)
        judge = chat_model_cls(judge_model_id, device=device, enable_thinking=False)
        label_token_ids = resolve_label_token_ids(judge.tokenizer, STANCE_LABELS)

        for source_dir in dirs_for_model:
            dest_path = source_dir / out_name
            rows = score_file(judge, source_dir / "trajectories.jsonl", dest_path, label_token_ids)
            n_matches = sum(1 for row in rows if row.get("stance_label_argmax") == row.get("stance_label"))
            mass_values = [row["label_mass_total"] for row in rows if "label_mass_total" in row]
            manifest[str(source_dir)] = {
                "source": str(source_dir / "trajectories.jsonl"),
                "dest": str(dest_path),
                "n_rows": len(rows),
                "n_argmax_matches_stance_label": n_matches,
                "judge_model": judge_model_id,
                "median_label_mass_total": float(np.median(mass_values)) if mass_values else None,
            }
            logger.info("{}: {} rows, {}/{} argmax matches stance_label", source_dir, len(rows), n_matches, len(rows))
    return manifest
