"""The S3 closed loop: generate a turn, read the state, choose the next action.

The difference from `run_schedule_arm` that matters is WHERE the action comes
from. A schedule arm knows every action before the first token is generated; a
closed-loop arm cannot know the action at turn t until turn t-1 has been
generated AND judged, so the in-loop judge sits inside the generation loop and
the arm's wall-clock is generation plus judging, not generation alone.

TWO JUDGES, AND THEY NEVER MEET. The judge called here is the agent's OWN
model (plan section 8): it is a SELECTION signal, it decides what the
controller does, and no number it produces is ever reported. The reporting
readout is a separate, later job with the independent judge that S0 calibrated.
Its verdicts are written into the rows as `followed_inloop` under that name so
the two can never be confused by a downstream reader, and
`score_sequor_trajectories.py` writes its own field.

WHAT A BLIND TURN DOES. The in-loop judge fails to parse on about 5% of calls
at cap 1024 (job 15739196), and `graded_readout`'s rule is that any parse
failure nulls the whole turn -- so the controller is state-blind on roughly one
turn in twenty. It does NOT then guess: a blind turn plays `none`, spends
nothing, and is counted in `n_blind_turns`. Carrying the previous state forward
instead would let the controller act on a stale reading and would make the
blindness invisible in the artifact, which is the failure mode the counter
exists to prevent.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .sequor_bank import SequorItem, system_message, targeted_constraint_block, user_message
from .sequor_constraint_judge import graded_readout
from .sequor_controllers import NONE, action_cost, greedy_action, mpc_action
from .sequor_trajectory import BranchArmConfig, _row

GenerateBatch = Callable[[list[list[dict]]], list[dict]]
JudgeBatch = Callable[[list[tuple[str, str]]], list[bool | None]]


def closed_loop_action(policy: str, item: SequorItem, state: tuple[bool, ...] | None,
                       tokens_left: int, turn: int, table: dict | None,
                       schedule: Sequence[int] | None, count_tokens) -> tuple[int, ...]:
    """One decision. Turn 1 states the constraints, so it takes no action in
    any arm -- the same rule the schedule arms follow, and the reason a branch
    arm has 19 pairs per item rather than 20."""

    if turn == 1:
        return NONE
    if policy in ("best_fixed_schedule", "equal_cost_random"):
        if not schedule[turn - 1]:
            return NONE
        everything = tuple(range(len(item.constraints)))
        return everything if action_cost(item, everything, count_tokens) <= tokens_left else NONE
    if state is None:  # the in-loop judge did not parse this turn; act on nothing
        return NONE
    if policy == "koopman_mpc":
        return mpc_action(table, turn, state, tokens_left)
    if policy == "greedy_targeted":
        return greedy_action(item, state, tokens_left, count_tokens)
    raise ValueError(f"unknown policy {policy!r}")


def run_closed_loop_arm(
    items: list[SequorItem], generate_batch: GenerateBatch, judge_batch: JudgeBatch,
    config: BranchArmConfig, run_id: str, policy: str, budgets: dict[str, int],
    tables: dict[str, dict], schedules: dict[str, list[int]], count_tokens,
) -> list[dict]:
    """One sustained trajectory per item under one policy, all items in lockstep.

    Lockstep across ITEMS rather than across turns is what keeps the batch wide:
    every item is at the same turn at the same time, so one generation call and
    one judging call serve the whole arm per turn. Trajectories never interact,
    so this is only a scheduling choice -- but it is the difference between 20
    wide batches and 800 narrow ones.
    """

    histories: list[list[dict]] = [
        [{"role": "system", "content": system_message(item)}] if config.constraints_in_system
        else ([{"role": "system", "content": config.system_prompt}] if config.system_prompt else [])
        for item in items
    ]
    states: list[tuple[bool, ...] | None] = [tuple([True] * len(item.constraints)) for item in items]
    tokens_left = [budgets[item.conversation_id] for item in items]
    prev_agent: list[str | None] = [None] * len(items)
    blind_turns = [0] * len(items)
    rows: list[dict] = []

    for turn in range(1, config.n_turns + 1):
        actions = [
            closed_loop_action(policy, item, states[i], tokens_left[i], turn,
                               tables.get(item.conversation_id),
                               schedules.get(item.conversation_id), count_tokens)
            for i, item in enumerate(items)
        ]
        messages = []
        for item, action in zip(items, actions):
            base = user_message(item, turn - 1, remind=False,
                                constraints_in_system=config.constraints_in_system)
            messages.append(base if not action
                            else f"{base}\n\n{targeted_constraint_block(item, action)}")
        prompts = [h + [{"role": "user", "content": m}] for h, m in zip(histories, messages)]
        generated = generate_batch(prompts)

        verdicts = judge_batch([
            (constraint, out["text"])
            for item, out in zip(items, generated) for constraint in item.constraints
        ])

        cursor = 0
        for i, (item, message, out, action) in enumerate(zip(items, messages, generated, actions)):
            k = len(item.constraints)
            followed = verdicts[cursor:cursor + k]
            cursor += k
            y_inloop, n_failed = graded_readout(followed)
            cost = action_cost(item, action, count_tokens) if action else 0
            if cost > tokens_left[i]:
                raise ValueError(
                    f"{item.conversation_id} turn {turn}: action costs {cost} with {tokens_left[i]} "
                    f"left. The policy returned an action the budget cannot cover.")
            tokens_left[i] -= cost

            row = _row(item, turn, policy, histories[i], message, out, prev_agent[i], config,
                       run_id, u_remind=int(bool(action)))
            row["inserted_tokens"] = cost
            row["action_named"] = list(action)
            row["n_named"] = len(action)
            row["tokens_left_after"] = tokens_left[i]
            row["budget"] = budgets[item.conversation_id]
            row["followed_inloop"] = list(followed)
            row["y_inloop"] = y_inloop
            row["n_inloop_parse_failures"] = n_failed
            row["state_before_action"] = None if states[i] is None else list(states[i])
            rows.append(row)

            if y_inloop is None:
                blind_turns[i] += 1
                states[i] = None
            else:
                states[i] = tuple(bool(b) for b in followed)
            histories[i] = histories[i] + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": out["text"]},
            ]
            prev_agent[i] = out["text"]

    for i, item in enumerate(items):
        for row in rows:
            if row["item_id"] == item.conversation_id:
                row["n_blind_turns"] = blind_turns[i]
    return rows


def budget_accounting(rows: list[dict], policies: Sequence[str]) -> dict:
    """What each arm actually spent, which is the claim "equal cost" rests on.

    Reported per arm rather than pooled: an arm that underspends its budget is
    not cheating, but a comparison that calls itself equal-cost while one arm
    spent two thirds of another's tokens is describing a different experiment
    than the one it ran.
    """

    out = {}
    for policy in policies:
        arm = [row for row in rows if row["branch"] == policy]
        if not arm:
            continue
        budgets = {(row["item_id"], row["seed"]): row["budget"] for row in arm}
        spent: dict[tuple, int] = {}
        for row in arm:
            key = (row["item_id"], row["seed"])
            spent[key] = spent.get(key, 0) + row["inserted_tokens"]
        out[policy] = {
            "n_rows": len(arm),
            "tokens_spent_total": sum(spent.values()),
            "tokens_budgeted_total": sum(budgets.values()),
            "share_of_budget_spent": sum(spent.values()) / sum(budgets.values()),
            "n_actions": sum(1 for row in arm if row["n_named"]),
            "n_targeted_actions": sum(1 for row in arm if 0 < row["n_named"] < len(row["constraints"])),
            "n_blanket_actions": sum(1 for row in arm if row["n_named"] == len(row["constraints"])),
            "n_blind_turns": sum(1 for row in arm if row["y_inloop"] is None),
            "blind_turn_share": sum(1 for row in arm if row["y_inloop"] is None) / len(arm),
            "overspent_items": sorted(k for k, v in spent.items() if v > budgets[k]),
        }
    return out
