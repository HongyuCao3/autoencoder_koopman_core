"""Verify that the `core` collection protocol can be reconstructed from the landed
trajectories, so a second backbone can be driven through the identical protocol.

The `core` datasets (`datasets/scalar/*`, `datasets/vector/*`) were supplied as a
finished corpus; no generator for them exists in this repository (see
`docs/protocols/DATA_COLLECTION_PROTOCOL.md`, which states the collection differs
from the collaborator's). Replaying the protocol against a new backbone therefore
requires recovering three things, each of which this script checks against the
landed rows rather than assuming:

  1. the turn-structure convention (what the model actually sees at turn t),
  2. the readout scorer (raw_generation -> measured value),
  3. the feedback rule (measured, target -> the next user message).

Only (2) and (3) need reconstruction: (1) is verified, and turn-1 prompts are
replayed verbatim from the landed rows.

`--verify` reports per-task fidelity. It is a report, not a pass/fail gate: a task
whose scorer reproduces below 1.0 is still replayable, but then the landed Qwen
column is NOT a valid control for it and a same-harness Qwen arm is mandatory.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]

SCALAR = REPO / "datasets" / "scalar"
VECTOR = REPO / "datasets" / "vector"

TASK_PATHS = {
    "sentence_length_t10": SCALAR / "sentence_length_t10",
    "character_length_t5": SCALAR / "character_length_t5",
    "average_word_length_t5": SCALAR / "average_word_length_t5",
    "sentiment_t5": SCALAR / "sentiment_t5",
    "formality_t5": SCALAR / "formality_t5",
    "even_odd_t5": SCALAR / "even_odd_t5",
    "vector_count_stage1_t10": VECTOR / "vector_count_stage1_t10",
    "vector_count_stage2_t10": VECTOR / "vector_count_stage2_t10",
}

# Recovered by fitting against every landed row, not from an example. The three
# splitting characters and the hyphen rule are each pinned by a residual that
# disappears only under that rule; see the module docstring of the campaign plan.
_WORD_SPLIT = re.compile(r"[\s/’]+")


def word_tokens(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT.split((text or "").strip()) if w]


def word_count(text: str) -> int:
    """`up-to-date` counts 2, `well-being` counts 1, `24/7` counts 2."""
    return sum(1 + w.count("-") // 2 for w in word_tokens(text))


def character_count(text: str) -> int:
    return len(text or "")


def average_word_length(text: str) -> float:
    words = word_tokens(text)
    if not words:
        return 0.0
    return sum(len(re.sub(r"[^A-Za-z0-9]", "", w)) for w in words) / len(words)


def comma_count(text: str) -> int:
    return (text or "").count(",")


def first_integer(text: str):
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None


# field on the row -> function of raw_generation that must reproduce it
SCORERS = {
    "sentence_length_t10": {"measured_raw_count": word_count},
    "character_length_t5": {"measured_raw_count": character_count},
    "average_word_length_t5": {"measured_raw_value": average_word_length},
    "even_odd_t5": {"integer_value": first_integer},
    "vector_count_stage1_t10": {"word_count": word_count,
                                "average_word_length": average_word_length},
    "vector_count_stage2_t10": {"word_count": word_count,
                                "average_word_length": average_word_length,
                                "comma_count": comma_count},
    # model-based readouts: reproduced by the external scorer, not by arithmetic
    "formality_t5": {},
    "sentiment_t5": {},
}

EXTERNAL_SCORERS = {
    "formality_t5": "hf_lendiglearn_mdeberta_v3_formality_isotonic",
    "sentiment_t5": "hf_cardiff_roberta_latest",
}


def load(task: str) -> list[dict]:
    path = TASK_PATHS[task] / "trajectories.jsonl"
    return [json.loads(line) for line in path.open()]


def by_trajectory(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["trajectory_id"], {})[r["turn"]] = r
    return out


def check_turn_structure(rows: list[dict]) -> dict:
    """The three conventions a replay harness depends on."""
    traj = by_trajectory(rows)
    pairs = prefix_ok = assistant_is_raw = feedback_is_next_user = 0
    for turns in traj.values():
        for t in sorted(turns):
            if t + 1 not in turns:
                continue
            pairs += 1
            cur, nxt = turns[t]["prompt_messages"], turns[t + 1]["prompt_messages"]
            if nxt[: len(cur)] == cur:
                prefix_ok += 1
            if nxt[-2]["content"] == (turns[t].get("raw_generation") or "\0"):
                assistant_is_raw += 1
            if nxt[-1]["content"] == (turns[t].get("feedback_text") or "\0"):
                feedback_is_next_user += 1
    return {"pairs": pairs, "history_is_append_only": prefix_ok,
            "assistant_turn_is_raw_generation": assistant_is_raw,
            "user_turn_is_feedback_of_previous_row": feedback_is_next_user}


def check_scorers(task: str, rows: list[dict]) -> dict:
    out = {}
    for field, fn in SCORERS[task].items():
        n = hit = 0
        for r in rows:
            want = r.get(field)
            if want is None:
                continue
            n += 1
            got = fn(r.get("raw_generation"))
            if isinstance(want, float) or isinstance(got, float):
                ok = got is not None and abs(float(got) - float(want)) < 1e-9
            else:
                ok = got == want
            hit += bool(ok)
        out[field] = {"n": n, "exact": hit, "fidelity": (hit / n) if n else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASK_PATHS))
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    report = {}
    for task in args.tasks:
        rows = load(task)
        report[task] = {
            "n_rows": len(rows),
            "n_trajectories": len(by_trajectory(rows)),
            "turn_structure": check_turn_structure(rows),
            "scorers": check_scorers(task, rows),
            "external_scorer": EXTERNAL_SCORERS.get(task),
        }

    for task, rep in report.items():
        ts = rep["turn_structure"]
        p = ts["pairs"]
        struct_ok = all(ts[k] == p for k in
                        ("history_is_append_only", "assistant_turn_is_raw_generation",
                         "user_turn_is_feedback_of_previous_row"))
        print(f"== {task}  rows={rep['n_rows']} traj={rep['n_trajectories']}")
        print(f"   turn structure {p}/{p} on all three conventions: "
              f"{'OK' if struct_ok else 'MISMATCH ' + str(ts)}")
        if rep["external_scorer"]:
            print(f"   readout is an external model: {rep['external_scorer']} "
                  f"(not reproducible by arithmetic; must be pinned separately)")
        for field, s in rep["scorers"].items():
            print(f"   scorer {field:24s} {s['exact']:6d}/{s['n']:<6d} "
                  f"fidelity={s['fidelity']:.4f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
