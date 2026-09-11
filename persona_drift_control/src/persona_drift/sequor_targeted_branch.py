"""Targeted-reminder counterfactual branches for the `constraint` line
(docs/experiments/constraint_results.md, the option (a)/(b) close-out).

WHY THIS ARM EXISTS. S3's `koopman_mpc` arm died twice on paper: the optimal
SCHEDULE is the same for every trajectory under a scalar operator (2026-09-10)
and under a per-constraint binary state, and it stays the same when the
objective is changed to a threshold with priced reminders (2026-09-11). The
mechanism behind both is that the action is a single switch -- "restate all
three constraints or not" -- so the state can only scale what a reminder is
worth, never change which turn is the best one to spend it on. This arm changes
the ACTION instead: restate only the constraint that broke. Which constraint
that is cannot be scheduled in advance, so if a targeted reminder is more
effective than a blanket one, the closed loop has something to do that no fixed
plan can imitate.

WHAT IT GENERATES. Nothing new for the u=0 side: the branches hang off the
zero_control trajectories of `outputs/sequor_s1_arm/`, whose turn-t response
was generated from exactly the prefix these branches use, and is therefore
already the no-reminder counterfactual. Each branch point adds two generations
-- blanket (the whole block, the action S1 measured) and targeted (only the
violated constraints) -- and the prefix identity is not assumed but checked:
every rebuilt history is hashed and compared against the `prefix_sha256` the
stored row recorded when it was generated.

SELECTION VS REPORT. The branch point is chosen by the verdict on turn t-1 and
the outcome is read on turn t, so no number is selected and reported on the
same observation. Conditioning on "the judge called this constraint broken at
t-1" does select for the judge's own false negatives, but all three branches
are selected on the identical condition and share the prefix, so the CONTRAST
is unaffected -- only the levels are. The outcome numbers still come from a
fresh independent-judge pass, never from the readout that chose the target.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

from .sequor_bank import SequorItem, system_message, targeted_constraint_block, user_message
from .sequor_trajectory import BranchArmConfig, _row, prefix_digest

GenerateBatch = Callable[[list[list[dict]], list[int]], list[dict]]

BLANKET = "blanket_remind"
TARGETED = "targeted_remind"
BASE_ARM = "zero_control"


def base_trajectories(rows: list[dict], arm: str = BASE_ARM) -> dict[tuple[str, int], list[dict]]:
    """The stored rows of one arm, grouped per (item, seed) and ordered by turn."""

    out: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row["branch"] != arm:
            continue
        out.setdefault((row["item_id"], row["seed"]), []).append(row)
    for key, turns in out.items():
        turns.sort(key=lambda r: r["turn"])
        expected = list(range(1, len(turns) + 1))
        if [r["turn"] for r in turns] != expected:
            raise ValueError(f"{key}: turns are {[r['turn'] for r in turns]}, expected {expected}")
    return out


def rebuild_history(item: SequorItem, turns: list[dict], upto_turn: int,
                    config: BranchArmConfig) -> list[dict]:
    """The conversation prefix turn `upto_turn` was generated from.

    Rebuilt from the stored rows rather than re-simulated: the user messages
    and the model's own replies are in the artifact, so nothing here depends on
    reproducing a sampled generation. The caller checks the result against the
    stored digest, which is what makes "the same prefix" a fact rather than an
    assumption.
    """

    history: list[dict] = []
    if config.constraints_in_system:
        history.append({"role": "system", "content": system_message(item)})
    elif config.system_prompt:
        history.append({"role": "system", "content": config.system_prompt})
    for row in turns:
        if row["turn"] >= upto_turn:
            break
        history.append({"role": "user", "content": row["user_message"]})
        history.append({"role": "assistant", "content": row["agent_message"]})
    return history


def violated_indices(followed: Sequence[bool | None] | None) -> list[int] | None:
    """Which constraints the readout says are broken. `None` when the turn is
    unusable: a turn with any parse failure has no state, the same rule
    `graded_readout` applies -- acting on a partially parsed turn would let the
    target depend on which verdict failed to parse."""

    if followed is None or any(b is None for b in followed):
        return None
    return [i for i, b in enumerate(followed) if not b]


def branch_points(base: dict[tuple[str, int], list[dict]], verdicts: dict[tuple[str, int], list],
                  n_turns: int) -> list[dict]:
    """Every (item, seed, turn) where a targeted reminder would be a distinct
    action: turn >= 2, the previous turn parsed, and at least one constraint
    broken. A turn with nothing broken is dropped because the targeted and
    blanket actions are not comparable there -- there is no target."""

    points = []
    for (item_id, seed), turns in sorted(base.items()):
        for row in turns:
            turn = row["turn"]
            if turn < 2 or turn > n_turns:
                continue
            previous = verdicts.get((row["trajectory_id"], turn - 1))
            broken = violated_indices(previous)
            if not broken:  # None (unparsed) or [] (nothing to target)
                continue
            points.append({
                "item_id": item_id, "seed": seed, "turn": turn,
                "violated_indices": broken,
                "n_violated_at_t_minus_1": len(broken),
                "trajectory_id": row["trajectory_id"],
            })
    return points


def run_targeted_branches(items: dict[str, SequorItem], base: dict[tuple[str, int], list[dict]],
                          points: list[dict], generate_batch: GenerateBatch,
                          config: BranchArmConfig, run_id: str) -> list[dict]:
    """Two generations per branch point, both from the stored prefix.

    Batched over branch points rather than over turns: these are one-step
    counterfactuals that never continue, so nothing has to run in lockstep and
    the whole arm is one wide batch per action.
    """

    prompts_blanket, prompts_targeted, meta = [], [], []
    for point in points:
        item = items[point["item_id"]]
        turns = base[(point["item_id"], point["seed"])]
        row = turns[point["turn"] - 1]
        history = rebuild_history(item, turns, point["turn"], config)
        digest = prefix_digest(history)
        if digest != row["prefix_sha256"]:
            raise ValueError(
                f"{point['item_id']} seed {point['seed']} turn {point['turn']}: rebuilt prefix "
                f"hashes to {digest[:12]} but the stored row recorded {row['prefix_sha256'][:12]}. "
                f"The branch would not be a counterfactual of this trajectory.")
        expected_user = user_message(item, point["turn"] - 1, remind=False,
                                     constraints_in_system=config.constraints_in_system)
        if expected_user != row["user_message"]:
            raise ValueError(
                f"{point['item_id']} turn {point['turn']}: the bank's user message differs from the "
                f"one the arm stored; the bank and the artifact have drifted apart")
        blanket = f"{row['user_message']}\n\n{item.constraint_block}"
        targeted = f"{row['user_message']}\n\n{targeted_constraint_block(item, point['violated_indices'])}"
        prompts_blanket.append(history + [{"role": "user", "content": blanket}])
        prompts_targeted.append(history + [{"role": "user", "content": targeted}])
        meta.append((point, item, history, blanket, targeted, turns))

    # One sampling seed per branch point, shared by that point's two branches:
    # common random numbers, so the targeted-minus-blanket difference is not the
    # difference of two draws. The seed is the stored trajectory's own, which is
    # also what labels the row -- a branch off the seed-2 trajectory that
    # recorded seed 0 would be paired with the wrong u=0 row downstream.
    seeds = [point["seed"] for point in points]
    out_blanket = generate_batch(prompts_blanket, seeds)
    out_targeted = generate_batch(prompts_targeted, seeds)

    rows = []
    for (point, item, history, blanket, targeted, turns), gen_b, gen_t in zip(
            meta, out_blanket, out_targeted):
        previous_reply = turns[point["turn"] - 2]["agent_message"]
        for branch, message, generated in ((BLANKET, blanket, gen_b), (TARGETED, targeted, gen_t)):
            row = _row(item, point["turn"], branch, history, message, generated, previous_reply,
                       replace(config, variant=branch, seed=point["seed"]), run_id,
                       u_remind=1, per_turn_branch=True)
            row["violated_indices"] = point["violated_indices"]
            row["n_violated_at_t_minus_1"] = point["n_violated_at_t_minus_1"]
            row["targeted_constraints"] = (
                [item.constraints[i] for i in point["violated_indices"]] if branch == TARGETED
                else list(item.constraints))
            row["base_trajectory_id"] = point["trajectory_id"]
            rows.append(row)
    return rows


def targeting_agreement(primary: dict[tuple[str, int], list], secondary: dict[tuple[str, int], list],
                        points: list[dict]) -> dict:
    """What a cheaper in-loop judge would have targeted instead, at zero GPU.

    The pilot targets on one readout; a real controller would run the other.
    This reports how often the two pick the same constraint set on the same
    turns, so the discount a 4B in-loop judge costs is measured rather than
    assumed -- together with the turns it would have skipped entirely, which is
    where its 5% parse-failure rate lands.
    """

    same, different, would_skip, would_act_on_nothing = 0, 0, 0, 0
    for point in points:
        other = violated_indices(secondary.get((point["trajectory_id"], point["turn"] - 1)))
        if other is None:
            would_skip += 1
        elif not other:
            would_act_on_nothing += 1
        elif other == point["violated_indices"]:
            same += 1
        else:
            different += 1
    total = len(points) or 1
    return {
        "n_branch_points": len(points),
        "same_target": same, "different_target": different,
        "would_have_skipped_unparsed": would_skip,
        "would_have_seen_nothing_broken": would_act_on_nothing,
        "agreement_share": same / total,
        "reading": "the share a controller reading the other judge would have targeted identically; "
                   "the skipped and nothing-broken counts are turns it would not have acted on at all",
    }
