"""The action space and prompt template for the `tsar_cefr` open-loop arm
(plan section 4.1, docs/operational_plan_tsar_cefr_line_2026-09-13.md).

FOUR ACTIONS, DRAWN INDEPENDENTLY OF THE STATE. This is the excitation arm:
the point is identification data, so `u_t` must not be a function of anything
the operator will later be asked to predict from. They are drawn uniformly, and
the draw is seeded by (seed, text_id) so a trajectory can be replayed without
replaying the whole arm.

`copy` IS AN ACTION AND IT IS GENERATED, not spliced. It is the zero-input
control that identifies `A` -- the same role S1b's antithetic arm plays on the
`constraint` line -- and the quantity it has to measure is what the rewriting
harness does when it is told to do nothing, which is not the same thing as what
a `cp` does. The share of steps where the model returned its input verbatim is
recorded (`unchanged_verbatim`); it is a property of the harness worth knowing,
not a defect.

THE TEMPLATE IS FIXED AND ONLY THE ACTION PHRASE AND THE TARGET LEVEL MOVE.
A template that varied with the step, the source or the current level would
put information into `u` that the operator is supposed to have to infer from
the state.

CEFR wording note: the phrases never name a numeric level to move TO. `ell` is
a readout, not something the prompt should anchor; telling the model "go to
B1" on every step would make `step_down` and `half_step_down` the same action
whenever the text is already near B1, and the design needs them separable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TARGET_LEVELS = ("A2", "B1")


@dataclass(frozen=True)
class Action:
    name: str
    phrase: str


ACTIONS = (
    Action("step_down",
           "Rewrite the text so that it is one CEFR level simpler than it is now."),
    Action("half_step_down",
           "Rewrite the text so that it is slightly simpler than it is now."),
    Action("paraphrase",
           "Rewrite the text using different wording, keeping it at the CEFR level it is now."),
    Action("copy",
           "Return the text exactly as it is, with no changes."),
)

ACTION_NAMES = tuple(a.name for a in ACTIONS)
ZERO_INPUT_ACTION = "copy"

SYSTEM_PROMPT = (
    "You rewrite English text for readability. "
    "Reply with the rewritten text only: no preface, no explanation, no quotation marks."
)


def action_by_name(name: str) -> Action:
    for action in ACTIONS:
        if action.name == name:
            return action
    raise KeyError(f"unknown action {name!r}; the space is {ACTION_NAMES}")


def user_message(text: str, action: str, target_cefr: str) -> str:
    """The step prompt. `text` is the PREVIOUS step's output, not the source."""
    if target_cefr not in TARGET_LEVELS:
        raise ValueError(f"unexpected target_cefr {target_cefr!r}; expected one of {TARGET_LEVELS}")
    return (
        f"The reader you are writing for is at CEFR level {target_cefr}.\n"
        f"{action_by_name(action).phrase}\n\n"
        f"Text:\n{text}\n\n"
        "Rewritten text:"
    )


def saturation_message(text: str, target_cefr: str) -> str:
    """The D-0 probe (plan ruling 4): one shot straight at the target level.

    If the model lands on the target level in one move on more than 80% of
    items there is no regulation problem left for a controller to solve, and
    D-0 fires. It is a DIFFERENT prompt from the step template on purpose --
    it is allowed to name the level, because naming it is the whole probe.
    """
    if target_cefr not in TARGET_LEVELS:
        raise ValueError(f"unexpected target_cefr {target_cefr!r}; expected one of {TARGET_LEVELS}")
    return (
        f"Rewrite the text so that a reader at CEFR level {target_cefr} can read it, "
        f"and so that it is not simpler than CEFR level {target_cefr}.\n\n"
        f"Text:\n{text}\n\n"
        "Rewritten text:"
    )


def draw_actions(seed: int, text_id: str, n_steps: int) -> list[str]:
    """Uniform over the four actions, independent of state, replayable.

    Seeded by (seed, text_id) rather than by a single stream so that adding an
    item or reordering the bank does not change any other item's actions.
    """
    rng = random.Random(f"tsar_cefr|{seed}|{text_id}")
    return [rng.choice(ACTION_NAMES) for _ in range(n_steps)]
