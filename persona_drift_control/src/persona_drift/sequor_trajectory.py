"""The `constraint` line's S0-0 counterfactual-branch arm
(docs/experiments/constraint_signal_screening.md section 2).

Every turn t >= 2 is generated TWICE from a byte-identical prefix: once with
u=0 (which is the trajectory that continues) and once with u=1, the constraint
block restated. The one-step gain of a reminder is therefore MEASURED per
(item, turn) rather than regressed for -- the design that finally produced a
CI excluding 0 on the ERGO line, here placed FIRST instead of last.

Generation is a batched callable supplied by the caller, so this module holds
no serving stack and is testable without a GPU. The arm runs in LOCKSTEP over
turns (all items at turn t, then all at t+1), which is what makes the batches
big enough for vLLM to be worth its own environment, and is the "concurrency
across trajectories, not across turns" the plan asks for.

Judging is deliberately NOT here. The graded readout `y_t` is applied
afterwards, by scripts/score_sequor_trajectories.py, so the same responses can
be scored by both the in-loop (self) judge and the independent reporting judge
-- and so a judge swap never means re-generating trajectories, the coupling
that made an instrument fix cost the ERGO line 18 jobs (docs/LEDGER.md
section 2).
"""

from __future__ import annotations

import random

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from .sequor_bank import SequorItem, system_message, user_message

GenerateBatch = Callable[[list[list[dict]]], list[dict]]

BASE = "base"
REMINDED = "reminded"


@dataclass(frozen=True)
class BranchArmConfig:
    model_id: str
    n_turns: int = 20
    seed: int = 0
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    system_prompt: str | None = None
    constraints_in_system: bool = False
    variant: str | None = None


def word_jaccard(a: str, b: str) -> float:
    """Word-set overlap of two responses.

    The `constraint` line's echo record (plan section 2): a reminder invites the
    model to copy the constraint text back instead of obeying it, and this
    separates "echoed" from "recovered". ERGO measured the same thing with the
    same formula (`scripts/analyze_ergo_branch_pairs.py`,
    `scripts/fit_koopman_ergo_branch_lifted.py`); those two copies are left
    alone on purpose -- that line's artifacts are under analysis and its
    scripts are its reproduction path.
    """

    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def prefix_digest(conversation: Sequence[dict]) -> str:
    """SHA-256 of the conversation prefix a turn is generated from.

    The counterfactual premise is not assumed, it is recorded: both rows of a
    pair carry this digest, and the analysis refuses any pair whose two rows
    disagree (`assert_pairs_share_prefix`). A prefix that quietly differed
    would turn a prompt difference into a measured "causal gain".
    """

    payload = json.dumps(list(conversation), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row(
    item: SequorItem, turn: int, branch: str, prefix: Sequence[dict], message: str,
    generated: dict, prev_agent_message: str | None, config: BranchArmConfig, run_id: str,
    u_remind: int | None = None,
) -> dict:
    """`u_remind` defaults to the branch label, which is right for the
    counterfactual arm where the label IS the action. A schedule arm labels
    rows by the arm they belong to and takes an action per turn, so it passes
    `u` explicitly -- without that the reminded turns of `constant_remind`
    would be recorded as u=0 and every downstream `B` would be unidentifiable.
    """

    text = generated["text"]
    u = int(branch == REMINDED) if u_remind is None else int(u_remind)
    return {
        "run_id": run_id,
        "trajectory_id": f"{item.conversation_id}__{branch}__s{config.seed}__t{turn}"
        if branch == REMINDED else f"{item.conversation_id}__{branch}__s{config.seed}",
        "item_id": item.conversation_id,
        "tuple_id": item.tuple_id,
        "turn": turn,
        "branch": branch,
        "u_remind": u,
        "n_turns": config.n_turns,
        "constraints": list(item.constraints),
        "constraint_ids": list(item.constraint_ids),
        "constraint_block": item.constraint_block,
        "prefix_sha256": prefix_digest(prefix),
        "user_message": message,
        "agent_message": text,
        "finish_reason": generated.get("finish_reason"),
        "hit_token_cap": generated.get("finish_reason") == "length",
        "n_output_tokens": generated.get("n_output_tokens"),
        "inserted_chars": len(item.constraint_block) + 2 if u else 0,
        "inserted_tokens": generated.get("n_inserted_tokens"),
        "echo_jaccard_prev": None if prev_agent_message is None else word_jaccard(prev_agent_message, text),
        "echo_verbatim_prev": None if prev_agent_message is None else text.strip() == prev_agent_message.strip(),
        "model": config.model_id,
        "seed": config.seed,
        "system_prompt": config.system_prompt,
        "constraints_in_system": config.constraints_in_system,
        "variant": config.variant,
        "decoding_config": {
            "temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k,
            "max_new_tokens": config.max_new_tokens,
        },
    }


def run_branch_arm(
    items: list[SequorItem], generate_batch: GenerateBatch, config: BranchArmConfig, run_id: str,
    branch: bool = True,
) -> list[dict]:
    """Lockstep over turns; returns base rows and reminded (counterfactual) rows.

    Only the u=0 branch advances the conversation. The u=1 response is scored
    and discarded, which is what keeps every pair's prefix identical to the
    single base trajectory rather than to a diverging reminded trajectory --
    the arm measures the ONE-STEP gain of a reminder, and a second-order
    reminded history would be a different (and unpaired) quantity.

    `branch=False` runs the base trajectory alone -- the zero-control arm the
    upstream-fidelity check needs, where the question is only what the
    retention curve looks like under a given harness.

    Turn 1 has no reminded row: it already carries the constraint block, so
    u=1 there is not a distinct action (`sequor_bank.user_message` refuses it).
    With T=20 that gives 19 pairs per item, not 20 -- 228 pairs at N=12, where
    the screening plan's arithmetic said 240. The gates' thresholds are
    unchanged; their MDE is computed from the pairs that exist.
    """

    histories: list[list[dict]] = [
        [{"role": "system", "content": system_message(item)}] if config.constraints_in_system
        else ([{"role": "system", "content": config.system_prompt}] if config.system_prompt else [])
        for item in items
    ]
    prev_agent: list[str | None] = [None] * len(items)
    rows: list[dict] = []

    for turn in range(1, config.n_turns + 1):
        base_messages = [
            user_message(item, turn - 1, remind=False, constraints_in_system=config.constraints_in_system)
            for item in items
        ]
        base_prompts = [h + [{"role": "user", "content": m}] for h, m in zip(histories, base_messages)]
        base_out = generate_batch(base_prompts)

        if branch and turn > 1:
            reminded_messages = [
                user_message(item, turn - 1, remind=True, constraints_in_system=config.constraints_in_system)
                for item in items
            ]
            reminded_prompts = [h + [{"role": "user", "content": m}] for h, m in zip(histories, reminded_messages)]
            reminded_out = generate_batch(reminded_prompts)
            for item, history, message, generated, prev in zip(
                items, histories, reminded_messages, reminded_out, prev_agent
            ):
                rows.append(_row(item, turn, REMINDED, history, message, generated, prev, config, run_id))

        for i, (item, message, generated) in enumerate(zip(items, base_messages, base_out)):
            rows.append(_row(item, turn, BASE, histories[i], message, generated, prev_agent[i], config, run_id))
            histories[i] = histories[i] + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": generated["text"]},
            ]
            prev_agent[i] = generated["text"]

    return rows


SCHEDULE_ARMS = ("zero_control", "constant_remind", "bernoulli", "antithetic")


def build_schedules(items: list[SequorItem], n_turns: int, seed: int) -> dict[str, dict[str, list[int]]]:
    """The per-turn action for each S1 arm, as `schedules[arm][item_id][turn-1]`.

    Turn 1 is u=0 in EVERY arm. It states the constraints, so "remind" there is
    not a distinct action -- `sequor_bank.user_message` refuses it, and the
    same fact is why the branch arm has 19 pairs per item and not 20.

    `bernoulli` and `antithetic` are drawn from ONE stream per (seed, item) and
    the antithetic arm is its exact complement, so the two trajectories differ
    in every action from turn 2 on. That is the pairing plan section 4 asks for
    -- each (item, turn) cell has exactly one reminded side -- and it is why
    `B` is a matched-pair estimate rather than an i.i.d. pooled one. It is also
    why the endpoint dose contrast is zero by construction, so terminal
    authority has to come from the zero_control/constant_remind pair instead:
    the two groups do not substitute for each other.
    """

    schedules: dict[str, dict[str, list[int]]] = {arm: {} for arm in SCHEDULE_ARMS}
    for item in items:
        draws = random.Random(f"{seed}:{item.conversation_id}")
        coin = [0] + [draws.randint(0, 1) for _ in range(n_turns - 1)]
        schedules["zero_control"][item.conversation_id] = [0] * n_turns
        schedules["constant_remind"][item.conversation_id] = [0] + [1] * (n_turns - 1)
        schedules["bernoulli"][item.conversation_id] = coin
        schedules["antithetic"][item.conversation_id] = [0] + [1 - u for u in coin[1:]]
    return schedules


def assert_schedules_are_well_formed(schedules: dict[str, dict[str, list[int]]], n_turns: int) -> dict:
    """G-S1's structural half, checked before any GPU time is spent on it.

    Not a statistical gate -- these are properties of the design, and if one
    fails the arm was built wrong. Turn 1 must be an action-free turn in every
    arm; the antithetic pair must be an exact complement from turn 2 on; and
    every arm must cover the same items, because a `B` estimated on one item
    set and compared against a curve from another is not a paired comparison.
    """

    items = {arm: sorted(by_item) for arm, by_item in schedules.items()}
    if len({tuple(v) for v in items.values()}) != 1:
        raise ValueError(f"arms cover different item sets: { {a: len(v) for a, v in items.items()} }")
    for arm, by_item in schedules.items():
        for item_id, u in by_item.items():
            if len(u) != n_turns:
                raise ValueError(f"{arm}/{item_id}: {len(u)} turns, expected {n_turns}")
            if u[0] != 0:
                raise ValueError(f"{arm}/{item_id}: turn 1 carries u=1, which is not an action")
    for item_id, u in schedules["bernoulli"].items():
        anti = schedules["antithetic"][item_id]
        if any(a + b != 1 for a, b in zip(u[1:], anti[1:])):
            raise ValueError(f"antithetic/{item_id} is not the complement of bernoulli")
    pooled = [u for by_item in (schedules["bernoulli"],) for us in by_item.values() for u in us[1:]]
    return {
        "n_items": len(schedules["zero_control"]),
        "n_turns": n_turns,
        "bernoulli_u_mean": sum(pooled) / len(pooled) if pooled else None,
        "antithetic_is_exact_complement": True,
        "turn1_action_free": True,
    }


def run_schedule_arm(
    items: list[SequorItem], generate_batch: GenerateBatch, config: BranchArmConfig, run_id: str,
    arm: str, schedule: dict[str, list[int]],
) -> list[dict]:
    """One sustained trajectory per item, actions taken from `schedule`.

    The difference from `run_branch_arm` that matters: here the chosen action's
    response ADVANCES the conversation. The branch arm generates the reminded
    turn from the base prefix and throws it away, which is what keeps its pairs
    byte-identical and makes it a one-step measurement. S1 needs the opposite
    -- the trajectory the controller would actually produce, where a reminder
    at turn 5 is in the history at turn 6 -- because the endpoint of a
    sustained policy is the quantity S1a compares and the trajectory is what
    S2 fits an operator to.

    Consequence worth stating: rows from two different arms do NOT share a
    prefix and must never be paired at the row level. They are paired by item
    (S1a) or by the complementary action within a matched cell (S1b).
    """

    if arm not in SCHEDULE_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {SCHEDULE_ARMS}")
    histories: list[list[dict]] = [
        [{"role": "system", "content": system_message(item)}] if config.constraints_in_system
        else ([{"role": "system", "content": config.system_prompt}] if config.system_prompt else [])
        for item in items
    ]
    prev_agent: list[str | None] = [None] * len(items)
    rows: list[dict] = []

    for turn in range(1, config.n_turns + 1):
        actions = [schedule[item.conversation_id][turn - 1] for item in items]
        messages = [
            user_message(item, turn - 1, remind=bool(u),
                         constraints_in_system=config.constraints_in_system)
            for item, u in zip(items, actions)
        ]
        prompts = [h + [{"role": "user", "content": m}] for h, m in zip(histories, messages)]
        out = generate_batch(prompts)
        for i, (item, message, generated, u) in enumerate(zip(items, messages, out, actions)):
            rows.append(_row(item, turn, arm, histories[i], message, generated,
                             prev_agent[i], config, run_id, u_remind=u))
            histories[i] = histories[i] + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": generated["text"]},
            ]
            prev_agent[i] = generated["text"]

    return rows


CAP_CRITERION = 0.05


def cap_accounting(rows: list[dict], criterion: float = CAP_CRITERION) -> dict:
    """Per-item share of responses that hit the generation cap.

    Checked PER ITEM, not globally, because response length is a function of
    the constraint set: job 15756689 sat at 18.6% overall while two of its
    twelve items ("Write a longer dialog", "Write creatively as a story")
    accounted for 75 of 87 truncations and the other ten were under 13%. A
    global average hides exactly the failure that matters, and a truncated
    response can fail a constraint the model would have satisfied.
    """

    by_item: dict[str, list[dict]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
    by_item_stats = {
        item: {
            "n_rows": len(item_rows),
            "n_hit_token_cap": sum(1 for r in item_rows if r["hit_token_cap"]),
            "token_cap_share": sum(1 for r in item_rows if r["hit_token_cap"]) / len(item_rows),
            "output_tokens_median": sorted(r["n_output_tokens"] for r in item_rows)[len(item_rows) // 2],
            "constraints": item_rows[0]["constraints"],
        }
        for item, item_rows in sorted(by_item.items())
    }
    over = [i for i, v in by_item_stats.items() if v["token_cap_share"] > criterion]
    capped = sum(1 for r in rows if r["hit_token_cap"])
    tokens = sorted(r["n_output_tokens"] for r in rows)
    return {
        "cap_criterion": criterion,
        "n_hit_token_cap": capped, "token_cap_share": capped / len(rows),
        "token_cap_by_item": by_item_stats,
        "items_over_cap_criterion": over, "cap_criterion_pass": not over,
        "output_tokens_median": tokens[len(tokens) // 2], "output_tokens_max": tokens[-1],
    }


def assert_pairs_share_prefix(rows: list[dict]) -> int:
    """Every reminded row must share its prefix digest with the base row of the
    same (item, turn). Returns the number of pairs checked.

    Raises rather than reporting a rate: a contaminated pair is not a noisy
    measurement, it is a different measurement, and the whole point of the
    branch design is that this cannot be assumed.
    """

    base = {(r["item_id"], r["turn"], r["seed"]): r for r in rows if r["branch"] == BASE}
    pairs = 0
    for row in rows:
        if row["branch"] != REMINDED:
            continue
        # The seed is part of the key: under sampled decoding an arm runs the
        # same item several times, and pairing across seeds would compare two
        # different trajectories rather than one action.
        sibling = base.get((row["item_id"], row["turn"], row["seed"]))
        if sibling is None:
            raise ValueError(
                f"reminded row {row['item_id']} turn {row['turn']} seed {row['seed']} "
                f"has no base sibling")
        if sibling["prefix_sha256"] != row["prefix_sha256"]:
            raise ValueError(
                f"prefix mismatch at {row['item_id']} turn {row['turn']}: the pair was NOT "
                f"generated from a byte-identical prefix, so its delta is not a causal gain"
            )
        pairs += 1
    return pairs


def arm_config_record(config: BranchArmConfig, items: list[SequorItem]) -> dict:
    return {
        **asdict(config),
        "item_ids": [item.conversation_id for item in items],
        "tuple_ids": [item.tuple_id for item in items],
        "n_items": len(items),
        "expected_pairs": len(items) * (config.n_turns - 1),
    }
