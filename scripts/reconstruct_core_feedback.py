#!/usr/bin/env python3
"""Recover the `core` FEEDBACK RULE -- leg 3 of the collection protocol -- so a
second backbone can be driven through the identical loop.

`verify_core_protocol_reconstruction.py` (gate B0) covered legs 1 and 2: it
verified the turn structure and reproduced the SCORERS. It did not reconstruct
leg 3, the map (measured, target, topic) -> the next user message, and without
that leg no new backbone can be driven at all: turn 1 replays verbatim, but
turn 2 onwards needs a feedback string this repository does not have a
generator for.

The standard here is the one B0 used for the tokenizer rule: a builder counts as
recovered only if it reproduces the landed `feedback_text` BYTE-EXACTLY on every
landed row of that task. Anything less is inferring architecture from examples,
and the task does not enter the second-backbone group.

A second failure mode this reports separately: a BRANCH THAT NEVER FIRES IN THE
LANDED DATA cannot be recovered from it, however good the fidelity looks. A new
backbone that behaves differently will walk straight into it. `even_odd_t5` is
exactly that case -- all 96 landed feedback rows say "Your answer was correct",
so the wrong-answer branch has no witness.

Read-only: reads `datasets/`, writes nothing.
"""

from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
TASK_DIR = {
    "sentence_length_t10": REPO / "datasets" / "scalar" / "sentence_length_t10",
    "character_length_t5": REPO / "datasets" / "scalar" / "character_length_t5",
    "average_word_length_t5": REPO / "datasets" / "scalar" / "average_word_length_t5",
    "even_odd_t5": REPO / "datasets" / "scalar" / "even_odd_t5",
    "vector_count_stage1_t10": REPO / "datasets" / "vector" / "vector_count_stage1_t10",
}


def fb_sentence_length_t10(r: dict) -> str:
    m, t, topic = r["measured_raw_count"], r["target_raw_count"], r["topic"]
    if abs(m - t) <= 1:
        return (f"Your previous answer is within 1 word of the target. Keep approximately "
                f"the same length and write one fresh sentence about {topic} with exactly "
                f"{t} words. Output only the sentence.")
    side = "too short" if m < t else "too long"
    return (f"Your previous answer was {side}: I asked for exactly {t} words, and it had "
            f"{m} words. Please try again about {topic}. Output one sentence only with "
            f"exactly {t} words and no explanation.")


def fb_character_length_t5(r: dict) -> str:
    m, t = r["measured_raw_count"], r["target_raw_count"]
    phrase = ("It was the correct length." if m == t
              else "It was too long." if m > t else "It was too short.")
    return (f"Your previous output had {m} characters. The target was {t}. {phrase} "
            f"Please try again. Output only a string of exactly {t} characters, with no "
            f"quotes and no explanation.")


def fb_average_word_length_t5(r: dict) -> str:
    m, t = r["measured_raw_value"], r["target_raw_value"]
    phrase = ("It was close to the target." if r["within_tolerance"]
              else "It was too high." if m > t else "It was too low.")
    return (f"Your previous sentence had average word length {m:.2f}. The target was "
            f"{t:.2f}. {phrase} Please revise the sentence to move closer to the target. "
            f"Output only one sentence.")


def fb_even_odd_t5(r: dict) -> str:
    # The wrong-answer branch has NO witness in the landed data (see module
    # docstring); this builder therefore only covers `correct`.
    if not r["categorical_correct"]:
        raise NotImplementedError("even_odd_t5: wrong-answer branch is unwitnessed")
    return (f"Your answer was correct. I asked for a {r['target_category']} number, and "
            f"you gave {r['integer_value']}. Please try again. Output only the integer "
            f"and nothing else.")


def fb_vector_count_stage1_t10(r: dict) -> str:
    """Two dimensions, each with its own tolerance. Both tolerances are fitted
    against the landed rows, not assumed: word count is |m-t| <= 1 (witnessed
    at 1 -> True and 2 -> False), average word length is |m-t| <= 0.25 (the
    largest in-tolerance gap is exactly 0.25, the smallest out is 0.2593)."""
    wc, awl = r["word_count"], r["average_word_length"]
    twc, tawl = r["target_word_count"], r["target_avg_word_length"]
    wc_dir = ("keep similar length" if abs(wc - twc) <= 1
              else "make it shorter" if wc > twc else "make it longer")
    awl_dir = ("keep similar word choice" if abs(awl - tawl) <= 0.25
               else "use slightly shorter words" if awl > tawl
               else "use slightly longer words")
    return (f"Your previous sentence had {wc} words and average word length {awl:.2f} letters.\n"
            f"The targets were approximately {twc} words and {tawl:.2f} letters per word.\n\n"
            f"Revise the sentence to move closer to both targets:\n"
            f"- word count: {wc_dir}\n"
            f"- average word length: {awl_dir}\n\n"
            f"Keep the same topic: {r['topic']}. Avoid overcorrecting. Output only one sentence.")


BUILDERS = {
    "sentence_length_t10": fb_sentence_length_t10,
    "character_length_t5": fb_character_length_t5,
    "average_word_length_t5": fb_average_word_length_t5,
    "even_odd_t5": fb_even_odd_t5,
    "vector_count_stage1_t10": fb_vector_count_stage1_t10,
}


def load(task: str) -> list[dict]:
    path = TASK_DIR[task] / "trajectories.jsonl"
    return [json.loads(line) for line in path.open()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=sorted(TASK_DIR), action="append")
    ap.add_argument("--show-mismatches", type=int, default=2)
    args = ap.parse_args()
    tasks = args.task or sorted(BUILDERS)

    worst = 1.0
    for task in tasks:
        builder = BUILDERS.get(task)
        if builder is None:
            print(f"{task:26} NO BUILDER YET")
            worst = 0.0
            continue
        rows = [r for r in load(task) if r.get("feedback_text")]
        ok, shown = 0, 0
        for r in rows:
            try:
                got = builder(r)
            except NotImplementedError:
                got = None
            if got == r["feedback_text"]:
                ok += 1
            elif shown < args.show_mismatches:
                shown += 1
                print(f"  [{task}] want {r['feedback_text'][:110]!r}")
                print(f"  [{task}] got  {(got or '<unwitnessed branch>')[:110]!r}")
        share = ok / len(rows) if rows else 0.0
        worst = min(worst, share)
        print(f"{task:26} byte-exact {ok}/{len(rows)} = {share:.4f}"
              + ("  RECOVERED" if share == 1.0 else "  NOT RECOVERED"))
    print(f"\nworst task fidelity = {worst:.4f}; a task enters the second-backbone group "
          f"only at 1.0000")


if __name__ == "__main__":
    main()
