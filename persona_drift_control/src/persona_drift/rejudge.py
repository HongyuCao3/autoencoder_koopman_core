"""Offline re-scoring of already-collected attack trajectories with a
different judge model -- the defense line's version of the paired
independent-judge rerun the sycophancy line did in job 15487325
(docs/experiments/sycophancy_screening_pilot.md "追加分析").

Why offline rather than a rerun: every row of an attack trajectory already
stores the three strings the judge prompt is built from (`plain_query`,
`attacker_query`, `agent_message`) plus the `seed`/`turn` its judge seed was
derived from, so the judge call can be replayed against different weights
without generating a single agent token. That makes the pairing exact by
construction -- the sycophancy comparison had to *verify* that its two runs
produced byte-identical agent text (200/200) before it could treat the two
label sets as paired; here the same text is literally reused, so agent-side
sampling noise is zero by design and every difference is the judge.

What this deliberately does NOT do: re-run the controllers. `u_remind` and
`excitation_design` are copied through untouched, so for the reactive arms
(`threshold`, `koopman_mpc*`) the rejudged rows describe *what the arm did
under the original judge, re-measured*. The decisions themselves were made
from the self-judge's y_probe and cannot be undone offline -- see
docs/experiments/koopman_defense_pilot.md section 七. For the fixed-schedule
arms (`fixed_t*`, `periodic`, `zero_control`) the schedule is judge-
independent, so their rejudged rows are exactly what a rerun with the new
judge would have measured, up to agent sampling.

The original scores are kept on every rejudged row under `*_self_judge`
names rather than dropped: the paired comparison
(scripts/compare_safety_judge_runs.py) reads both columns off one file, and
a rejudged file that lost the original would make the bias unrecoverable
after the fact.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Callable, Iterable

from loguru import logger

from .chat_model import ChatModel, GenerationConfig
from .safety_judge import judge_safety_score

# trajectory_runner.run_reminder_gated_trajectory derives each judge call's
# seed as seed * 1_000_000 + turn * 100 + JudgeCall.seed_offset, and
# attack_trajectory.py registers the safety judge with seed_offset=2. Kept
# here as a named constant (with this comment) so a rejudge reproduces the
# original seed exactly instead of inventing a new one -- the judge decodes
# greedily (do_sample=False) so this is bookkeeping rather than sampling
# control, but a run whose seeds don't match the run it is compared against
# is one more thing a reader has to rule out.
SAFETY_JUDGE_SEED_OFFSET = 2

# Columns the rejudged row overwrites; the original value of each is kept
# under "<name>_self_judge". `y_probe` is the alias control.py's controllers
# read, kept in sync with y_safety by trajectory_runner.
_REJUDGED_FIELDS = ("y_safety", "y_probe", "judge_parse_failure", "judge_raw_output", "judge_model")


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (row["trajectory_id"], int(row["turn"]))


def judge_seed_for_row(row: dict[str, Any]) -> int:
    return int(row["seed"]) * 1_000_000 + int(row["turn"]) * 100 + SAFETY_JUDGE_SEED_OFFSET


def rejudge_row(judge: Any, row: dict[str, Any], config: GenerationConfig | None = None) -> dict[str, Any]:
    """One row, re-scored. Returns a new dict; `row` is not mutated."""

    config = config or GenerationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
    y_safety, parse_failure, raw_output = judge_safety_score(
        judge,
        row["plain_query"],
        row["attacker_query"],
        row["agent_message"],
        seed=judge_seed_for_row(row),
        config=config,
    )
    rejudged = dict(row)
    for field in _REJUDGED_FIELDS:
        if field in row:
            rejudged[f"{field}_self_judge"] = row[field]
    rejudged["y_safety"] = y_safety
    rejudged["y_probe"] = y_safety
    rejudged["judge_parse_failure"] = parse_failure
    rejudged["judge_raw_output"] = raw_output
    rejudged["judge_model"] = judge.model_id
    return rejudged


def pending_rows(source_rows: Iterable[dict[str, Any]], done_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of `source_rows` still to be judged, given what a previous
    (possibly killed) run already wrote.

    A row counts as done only if its rejudged copy carries the *same*
    `agent_message`: the arms are still being extended to 5 seeds while this
    runs, and an arm directory whose trajectories.jsonl was rewritten (e.g.
    screening_common.prepare_resumable_trajectories_file truncating a
    partial trajectory) must not silently keep a stale score attached to a
    different reply."""

    done = {row_key(row): row.get("agent_message") for row in done_rows}
    return [row for row in source_rows if done.get(row_key(row), object()) != row["agent_message"]]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Tolerates one torn *final* line, and only that: the arms are still
    being extended to 5 seeds while this runs, so a read can land between a
    write and its flush. A malformed line anywhere else means a corrupt file
    and still raises."""

    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            logger.warning("{}: ignoring a torn final line ({} chars), presumably a write in flight", path, len(line))
    return rows


def rejudge_file(
    judge: Any,
    source_path: pathlib.Path,
    dest_path: pathlib.Path,
    config: GenerationConfig | None = None,
    log_every: int = 25,
) -> list[dict[str, Any]]:
    """Re-scores every row of `source_path` into `dest_path`, appending as it
    goes (a killed job keeps everything already written; resubmitting the
    same command picks up the rest). Returns all rejudged rows, including
    ones carried over from an earlier run."""

    source_rows = load_jsonl(source_path)
    kept = [row for row in load_jsonl(dest_path) if row_key(row) in {row_key(r) for r in source_rows}]
    todo = pending_rows(source_rows, kept)
    kept = [row for row in kept if row_key(row) not in {row_key(r) for r in todo}]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Rewrite what is being kept (dropping rows whose source disappeared or
    # changed), then append the new ones.
    with dest_path.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    logger.info("{}: {} rows total, {} already rejudged, {} to do", source_path, len(source_rows), len(kept), len(todo))
    start = time.monotonic()
    with dest_path.open("a") as handle:
        for i, row in enumerate(todo, start=1):
            rejudged = rejudge_row(judge, row, config=config)
            handle.write(json.dumps(rejudged) + "\n")
            kept.append(rejudged)
            if i % log_every == 0 or i == len(todo):
                handle.flush()
                logger.info("  [{}/{}] {} (+{:.0f}s)", i, len(todo), row_key(row), time.monotonic() - start)
    return kept


def rejudge_dirs(
    arm_dirs: list[pathlib.Path],
    judge_model_id: str,
    out_subdir: str,
    device: str = "cuda",
    source_name: str = "trajectories.jsonl",
    chat_model_cls: Callable[..., Any] = ChatModel,
    config: GenerationConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Loads the judge once and re-scores each arm directory into
    `<arm>/<out_subdir>/trajectories.jsonl`. Returns a per-arm manifest."""

    present = [d for d in arm_dirs if (d / source_name).exists()]
    missing = [str(d) for d in arm_dirs if d not in present]
    for path in missing:
        logger.warning("{}/{} missing (arm not run yet), skipped", path, source_name)
    if not present:
        raise SystemExit(f"no arm directory has a {source_name} to rejudge")

    # enable_thinking=False on the instance as well as per judge call: this
    # process only ever uses it as a judge, and judge_safety_score pins the
    # per-call value anyway (see screening_common.load_agent_and_judge).
    logger.info("loading judge model {} on {}", judge_model_id, device)
    judge = chat_model_cls(judge_model_id, device=device, enable_thinking=False)

    manifest: dict[str, dict[str, Any]] = {}
    for arm_dir in present:
        dest_path = arm_dir / out_subdir / source_name
        rows = rejudge_file(judge, arm_dir / source_name, dest_path, config=config)
        n_changed = sum(1 for row in rows if row["y_safety"] != row.get("y_safety_self_judge"))
        manifest[str(arm_dir)] = {
            "source": str(arm_dir / source_name),
            "dest": str(dest_path),
            "n_rows": len(rows),
            "n_changed": n_changed,
            "judge_model": judge_model_id,
            "self_judge_model": rows[0].get("judge_model_self_judge") if rows else None,
            "agent_model": rows[0].get("model") if rows else None,
        }
        logger.info("{}: {} rows, {} scores changed", arm_dir, len(rows), n_changed)
    return manifest
