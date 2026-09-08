import pytest

from persona_drift.control import ConstantRemindController, ZeroControlController
from persona_drift.ergo_math_bank import GSM8KShardedItem
from persona_drift.ergo_math_trajectory import (
    ErgoMathTrajectoryConfig,
    _consolidated_stimulus,
    _incremental_stimulus,
    run_ergo_math_trajectory,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeChatModel:
    def __init__(self, model_id="fake-model", reply="Current answer: 3360", thinking="fake reasoning"):
        self.model_id = model_id
        self._reply = reply
        self._thinking = thinking
        self.calls = 0
        self.messages_seen: list[list[dict[str, str]]] = []
        self.tokenizer = FakeTokenizer()

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        self.calls += 1
        self.messages_seen.append([dict(m) for m in messages])
        if return_thinking:
            return self._reply, self._thinking
        return self._reply


def _entry(gold="3360"):
    return GSM8KShardedItem(
        item_id="ergo_GSM8K_1246",
        gold_answer=gold,
        shards=("shard one", "shard two", "shard three", "shard four"),
    )


def test_run_ergo_math_trajectory_produces_one_row_per_shard():
    agent = FakeChatModel()
    rows = run_ergo_math_trajectory(agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t1")
    assert len(rows) == 4
    assert [row["turn"] for row in rows] == [1, 2, 3, 4]
    assert all(row["num_shards"] == 4 for row in rows)
    assert all(row["y_task_success"] == 1.0 for row in rows)
    assert all(row["y_probe"] == row["y_task_success"] for row in rows)
    assert agent.calls == 4


def test_zero_control_never_resets_and_accumulates_history():
    agent = FakeChatModel()
    rows = run_ergo_math_trajectory(
        agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t2", controller=ZeroControlController()
    )
    assert all(row["u_reset"] == 0 for row in rows)
    assert all(row["excitation_design"] == "zero_control" for row in rows)
    # Turn 4's agent-facing history should still contain turn 1's shard text --
    # history was never wiped.
    last_call_messages = agent.messages_seen[-1]
    assert any("shard one" in m["content"] for m in last_call_messages if m["role"] == "user")
    # 4 user turns + 3 prior assistant replies + this turn's user message = 7
    assert len(last_call_messages) == 7


def test_constant_remind_controller_resets_every_turn_and_wipes_history():
    agent = FakeChatModel()
    rows = run_ergo_math_trajectory(
        agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t3", controller=ConstantRemindController()
    )
    assert all(row["u_reset"] == 1 for row in rows)
    assert all(row["excitation_design"] == "constant_remind" for row in rows)
    # Every reset turn should see exactly one user message (the consolidated
    # stimulus), never any prior assistant turns.
    for call_messages in agent.messages_seen:
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
    # The 3rd (index 2) reset turn's consolidated message should mention every
    # shard revealed so far (turns 1-3), not just the newest one.
    third_turn_text = agent.messages_seen[2][0]["content"]
    assert "shard one" in third_turn_text
    assert "shard two" in third_turn_text
    assert "shard three" in third_turn_text
    assert "shard four" not in third_turn_text


def test_incorrect_answer_scores_zero_and_flags_no_parse_failure():
    agent = FakeChatModel(reply="Current answer: 1")
    rows = run_ergo_math_trajectory(agent=agent, judge=agent, entry=_entry(gold="3360"), seed=0, trajectory_id="t4")
    assert all(row["y_task_success"] == 0.0 for row in rows)
    assert all(row["judge_parse_failure"] is False for row in rows)


def test_unparseable_reply_flags_parse_failure():
    agent = FakeChatModel(reply="I need more information before I can answer.")
    rows = run_ergo_math_trajectory(agent=agent, judge=agent, entry=_entry(), seed=0, trajectory_id="t5")
    assert all(row["judge_parse_failure"] is True for row in rows)
    assert all(row["y_task_success"] == 0.0 for row in rows)


def test_records_item_and_run_metadata_on_every_row():
    entry = _entry()
    rows = run_ergo_math_trajectory(agent=FakeChatModel(), judge=FakeChatModel(), entry=entry, seed=7, trajectory_id="t6")
    for row in rows:
        assert row["item_id"] == entry.item_id
        assert row["gold_answer"] == entry.gold_answer
        assert row["seed"] == 7
        assert row["trajectory_id"] == "t6"


# docs/experiments/two_task_success_plan.md section 2 E1: reset_mode tests.


def test_reset_mode_overwrite_wipes_history_to_length_one():
    agent = FakeChatModel()
    config = ErgoMathTrajectoryConfig(reset_mode="overwrite")
    run_ergo_math_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=0,
        trajectory_id="t7",
        controller=ConstantRemindController(),
        config=config,
    )
    # Existing (byte-for-byte-unchanged) behavior: every reset turn's
    # agent-facing history is exactly the one consolidated user message.
    for call_messages in agent.messages_seen:
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"


def test_reset_mode_append_grows_history_monotonically():
    agent = FakeChatModel()
    config = ErgoMathTrajectoryConfig(reset_mode="append")
    run_ergo_math_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=0,
        trajectory_id="t8",
        controller=ConstantRemindController(),
        config=config,
    )
    lengths = [len(call_messages) for call_messages in agent.messages_seen]
    # Every turn appends the consolidated message onto the prior history
    # (which already grew by the assistant reply), so per-turn history
    # length must strictly increase, never reset back down to 1.
    assert lengths == sorted(lengths)
    assert all(later > earlier for earlier, later in zip(lengths, lengths[1:]))
    assert lengths[0] == 1


def test_reset_mode_overwrite_and_append_produce_same_stimulus_text():
    agent_overwrite = FakeChatModel()
    agent_append = FakeChatModel()
    run_ergo_math_trajectory(
        agent=agent_overwrite,
        judge=agent_overwrite,
        entry=_entry(),
        seed=0,
        trajectory_id="t9a",
        controller=ConstantRemindController(),
        config=ErgoMathTrajectoryConfig(reset_mode="overwrite"),
    )
    run_ergo_math_trajectory(
        agent=agent_append,
        judge=agent_append,
        entry=_entry(),
        seed=0,
        trajectory_id="t9b",
        controller=ConstantRemindController(),
        config=ErgoMathTrajectoryConfig(reset_mode="append"),
    )
    # The consolidated stimulus text itself must not depend on reset_mode --
    # only where it lands in agent_history differs. Compare each turn's
    # newest user message (last user-role message on each call).
    for overwrite_messages, append_messages in zip(agent_overwrite.messages_seen, agent_append.messages_seen):
        overwrite_stimulus = [m for m in overwrite_messages if m["role"] == "user"][-1]["content"]
        append_stimulus = [m for m in append_messages if m["role"] == "user"][-1]["content"]
        assert overwrite_stimulus == append_stimulus


def test_invalid_reset_mode_raises_value_error():
    with pytest.raises(ValueError):
        ErgoMathTrajectoryConfig(reset_mode="clobber")


# --- prompt_profile (docs/experiments/ergo_fidelity_restoration_plan.md R1) ---


def test_prompt_profile_legacy_stimulus_text_is_byte_for_byte_unchanged():
    # G-R1-0's implementation lock: "legacy" must reproduce the exact strings
    # every existing output directory was produced with. Asserted against
    # literals here rather than against the module constant, so that editing
    # the constant cannot silently move the baseline with the test.
    assert _incremental_stimulus("shard one") == (
        "shard one\n\n"
        'Give your current best-guess final numeric answer to the math problem, even if you are '
        'not fully confident yet or do not have all the details, on its own line in this exact '
        'format: "Current answer: X" (a single number).'
    )
    assert _consolidated_stimulus(["shard one", "shard two"]) == (
        "Here is the math problem, given as a list of clues (all the information you have "
        "been given so far in this conversation):\n- shard one\n- shard two\n\n"
        'Give your current best-guess final numeric answer to the math problem, even if you are '
        'not fully confident yet or do not have all the details, on its own line in this exact '
        'format: "Current answer: X" (a single number).'
    )


def test_prompt_profile_upstream_puts_one_system_message_first_and_keeps_it_across_resets():
    agent = FakeChatModel()
    config = ErgoMathTrajectoryConfig(reset_mode="overwrite", prompt_profile="upstream")
    run_ergo_math_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=0,
        trajectory_id="t_up1",
        controller=ConstantRemindController(),
        config=config,
    )
    for call_messages in agent.messages_seen:
        systems = [m for m in call_messages if m["role"] == "system"]
        # Exactly one, and first -- an overwrite reset rebuilds the history from
        # the initial base, so dropping the system message there would silently
        # revert to the legacy profile from the first reset onward.
        assert len(systems) == 1
        assert call_messages[0]["role"] == "system"
        assert "Current answer: X" in call_messages[0]["content"]


def test_prompt_profile_upstream_user_turns_carry_no_answer_format_instruction():
    agent = FakeChatModel()
    config = ErgoMathTrajectoryConfig(reset_mode="append", prompt_profile="upstream")
    rows = run_ergo_math_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=0,
        trajectory_id="t_up2",
        controller=ZeroControlController(),
        config=config,
    )
    for row in rows:
        assert "Current answer" not in row["user_message"]
    assert rows[0]["user_message"] == "shard one"


def test_prompt_profile_upstream_consolidated_stimulus_carries_no_answer_format_instruction():
    consolidated = _consolidated_stimulus(["shard one", "shard two"], "upstream")
    assert "Current answer" not in consolidated
    assert consolidated.endswith("- shard one\n- shard two")
    assert _incremental_stimulus("shard one", "upstream") == "shard one"


def test_prompt_profile_is_recorded_on_every_row():
    agent = FakeChatModel()
    rows = run_ergo_math_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=0,
        trajectory_id="t_up3",
        config=ErgoMathTrajectoryConfig(prompt_profile="upstream"),
    )
    assert {row["prompt_profile"] for row in rows} == {"upstream"}
    legacy_rows = run_ergo_math_trajectory(
        agent=FakeChatModel(), judge=agent, entry=_entry(), seed=0, trajectory_id="t_up4"
    )
    assert {row["prompt_profile"] for row in legacy_rows} == {"legacy"}


def test_invalid_prompt_profile_raises_value_error():
    with pytest.raises(ValueError, match="prompt_profile"):
        ErgoMathTrajectoryConfig(prompt_profile="upstreem")


# ---------------------------------------------------------------------------
# EK-A counterfactual branching
# (docs/experiments/ergo_fidelity_restoration_plan.md section 4, gate G-EKA-0)
# ---------------------------------------------------------------------------


class ContentHashChatModel:
    """Reply depends on the exact messages AND the seed, so any perturbation of
    the prefix, the message order, or the seed changes the output. A constant-
    reply stub cannot tell "shares the prefix" from "happens to agree"."""

    def __init__(self, model_id="hash-model"):
        self.model_id = model_id
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[int, list[dict[str, str]]]] = []

    def generate(self, messages, seed, config=None, enable_thinking=None, return_thinking=False):
        import hashlib
        import json as _json

        snapshot = [dict(m) for m in messages]
        self.calls.append((seed, snapshot))
        h = hashlib.sha256((_json.dumps(snapshot, sort_keys=True) + f"|{seed}").encode()).hexdigest()[:8]
        text = f"working {h}\nCurrent answer: {int(h, 16) % 100}"
        return (text, f"think {h}") if return_thinking else text


def _branch(reset_mode="append", prompt_profile="upstream", controller=None):
    from persona_drift.ergo_math_trajectory import run_ergo_math_branch_trajectory

    agent = ContentHashChatModel()
    base, cf = run_ergo_math_branch_trajectory(
        agent=agent,
        judge=agent,
        entry=_entry(),
        seed=7,
        trajectory_id="t",
        config=ErgoMathTrajectoryConfig(reset_mode=reset_mode, prompt_profile=prompt_profile),
        controller=controller or ZeroControlController(),
    )
    return agent, base, cf


def test_branch_base_rows_are_identical_to_a_plain_run():
    """The whole design rests on the base branch being unperturbed: if
    generating the counterfactual changed the base trajectory, the pair would
    not be a counterfactual of anything."""
    for reset_mode in ("overwrite", "append"):
        for profile in ("legacy", "upstream"):
            cfg = ErgoMathTrajectoryConfig(reset_mode=reset_mode, prompt_profile=profile)
            plain_agent = ContentHashChatModel()
            plain = run_ergo_math_trajectory(
                agent=plain_agent, judge=plain_agent, entry=_entry(), seed=7,
                trajectory_id="t", config=cfg, controller=ZeroControlController(),
            )
            _, base, _cf = _branch(reset_mode, profile)
            assert base == plain, f"base branch perturbed under {reset_mode}/{profile}"


def test_counterfactual_shares_the_prefix_byte_for_byte():
    agent, base, cf = _branch()
    # two generate calls per turn: base first, then the counterfactual
    per_turn = [agent.calls[i:i + 2] for i in range(0, len(agent.calls), 2)]
    assert len(per_turn) == len(base)
    for (seed_b, msgs_b), (seed_c, msgs_c) in per_turn:
        assert msgs_b[:-1] == msgs_c[:-1], "prefix differs between base and counterfactual"
        assert msgs_b[-1] != msgs_c[-1], "the two actions produced the same user message"


def test_counterfactual_flips_the_action_every_turn():
    _agent, base, cf = _branch()
    assert len(cf) == len(base) == 4
    assert [c["u_reset"] for c in cf] == [1 - b["u_reset"] for b in base]
    assert [c["branch_turn"] for c in cf] == [b["turn"] for b in base]


def test_counterfactual_reuses_the_base_agent_seed_but_not_the_judge_seed():
    """Common random numbers: same sampling draws, different action."""
    agent, base, _cf = _branch()
    per_turn = [agent.calls[i:i + 2] for i in range(0, len(agent.calls), 2)]
    for turn, ((seed_b, _), (seed_c, _)) in enumerate(per_turn, start=1):
        assert seed_b == seed_c == 7 * 1_000_000 + turn * 100 + 1


def test_counterfactual_rows_carry_linkage_and_base_rows_stay_schema_clean():
    _agent, base, cf = _branch()
    plain_keys = set(base[0])
    for c in cf:
        assert c["is_counterfactual"] is True
        assert c["base_trajectory_id"] == "t"
        assert c["trajectory_id"] == f"t#cf_t{c['branch_turn']}"
        assert set(c) - plain_keys == {"is_counterfactual", "base_trajectory_id", "branch_turn", "base_u_reset"}
    for b in base:
        assert set(b) == plain_keys
        assert "is_counterfactual" not in b


def test_branch_costs_exactly_two_generations_per_turn():
    agent, base, _cf = _branch()
    assert len(agent.calls) == 2 * len(base)


def test_overwrite_counterfactual_rebuilds_from_the_initial_history():
    """Under overwrite a reset discards the history, so the counterfactual's
    message list must be the initial history plus one consolidated user turn --
    not the accumulated prefix."""
    agent, base, cf = _branch(reset_mode="overwrite", prompt_profile="upstream")
    per_turn = [agent.calls[i:i + 2] for i in range(0, len(agent.calls), 2)]
    for turn, (_base_call, (_seed, msgs_c)) in enumerate(per_turn, start=1):
        assert msgs_c[0]["role"] == "system"
        assert len(msgs_c) == 2, f"turn {turn}: overwrite counterfactual kept history"


def test_branch_with_a_resetting_base_flips_to_no_reset():
    _agent, base, cf = _branch(controller=ConstantRemindController())
    assert all(b["u_reset"] == 1 for b in base)
    assert all(c["u_reset"] == 0 for c in cf)
    assert all(c["base_u_reset"] == 1 for c in cf)
