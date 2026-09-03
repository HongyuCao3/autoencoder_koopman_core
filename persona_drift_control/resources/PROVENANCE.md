# Vendored resources

These files are vendored copies of third-party data used by the persona-drift
probe scoring functions, so that a compute node without internet access can
still import `hundred_system_prompts.py` and score probe responses.

| File | Source | Upstream commit/version | License |
|---|---|---|---|
| `hundred_system_prompts.py` | `Naomibas/llm-system-prompts-benchmark` on Hugging Face | sha `0e5379e1465827c1d00d7c4102b793e5be269e77` (fetched 2026-08-27) | Apache-2.0 |
| `word_lists/google-10000-english-usa.txt` | `first20hours/google-10000-english`, `google-10000-english-usa.txt` | `main` branch (fetched 2026-08-27; upstream `hundred_system_prompts.py` still pointed at the old `master` branch, which 404s) | see upstream repo |
| `word_lists/one-syllable-sorted-by-prevalence.txt` | `gautesolheim/25000-syllabified-words-list`, `1-syllable-sorted-by-frequency.txt` | `main` branch (fetched 2026-08-27; file was renamed and the `master` branch removed upstream since `hundred_system_prompts.py` was written) | see `LICENSE.md` in that repo |
| `safemtdata_attack_600.json` | `SafeMTData/SafeMTData` on Hugging Face, `SafeMTData/Attack_600.json` (ActorAttack, Ren et al., "Derail Yourself", arXiv 2410.10700) | sha `04af7bd0b6b6044e797e936d79674e348316b9b8` (fetched 2026-08-31) | MIT |
| `mtbench_questions.jsonl` | `lm-sys/FastChat` on GitHub, `fastchat/llm_judge/data/mt_bench/question.jsonl` (MT-Bench, Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023) | commit `b494d0c6b4e7935f1764f8439e75da3e66beccc7` (last touched that path; fetched 2026-09-01) | Apache-2.0 |
| `sycon_false_presuppositions.jsonl` | `JiseungHong/SYCON-Bench` on GitHub, `false-presuppositions-setting/data/{questions,presuppositions,corrections}.txt` + `push_back.csv` merged into one JSONL (SYCON-Bench, Hong et al., "Measuring Sycophancy of Language Models in Multi-turn Dialogues", EMNLP 2025 Findings, arXiv:2505.23840) | `master` branch (fetched 2026-09-02; upstream has no tagged release) | MIT (SYCON-Bench repo); the four source `.txt`/`.csv` files themselves trace to the CREPE dataset (`velocityCavalry/CREPE`) per SYCON-Bench's own `data/source.txt` -- not independently re-verified here, flagged for whoever revisits licensing before any external release |

`hundred_system_prompts.py` carries two small local patches relative to the
upstream file, documented in a comment at the top of the file itself:
`download_file()` now reads from the local `word_lists/` cache before
touching the network (with corrected URLs), and a missing comma in
`random_probes` that silently merged two probe questions into one has been
restored.

Re-fetch commands, if these ever need to be refreshed:

```bash
curl -s "https://huggingface.co/datasets/Naomibas/llm-system-prompts-benchmark/raw/main/hundred_system_prompts.py" -o hundred_system_prompts.py
curl -s "https://raw.githubusercontent.com/first20hours/google-10000-english/main/google-10000-english-usa.txt" -o word_lists/google-10000-english-usa.txt
curl -s "https://raw.githubusercontent.com/gautesolheim/25000-syllabified-words-list/main/1-syllable-sorted-by-frequency.txt" -o word_lists/one-syllable-sorted-by-prevalence.txt
curl -sL "https://huggingface.co/datasets/SafeMTData/SafeMTData/resolve/main/SafeMTData/Attack_600.json" -o safemtdata_attack_600.json
curl -s "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl" -o mtbench_questions.jsonl
```

`sycon_false_presuppositions.jsonl` is not a straight re-hosted file: it's a local merge of four
upstream files (`questions.txt`/`presuppositions.txt`/`corrections.txt`, one line per item, plus
`push_back.csv` with a `Question` column and four `Pushback_1..4` columns), joined by line/row
order after verifying the `push_back.csv` `Question` column matches `questions.txt` line-for-line
(0 mismatches across all 200 items, checked at fetch time). Each output row is
`{item_id, category: "false_presupposition", question, presupposition, correction,
pushback_turns: [4 strings]}` -- used as **fixed, pre-generated escalating-pushback content**
for `docs/task/SYCOPHANCY_DRIFT_TASK_FEASIBILITY.md`, loaded by `sycophancy_bank.py`. `question`
is turn 1 (neutral, no pushback yet); `pushback_turns` are turns 2-5, replayed verbatim, same
"fixed replay, not adaptive" design as `safemtdata_attack_600.json`. **Known data-quality note**:
20/200 items (10%) have all four `pushback_turns` identical verbatim (an upstream generation
artifact, not a bug in this merge) -- these are still valid sustained-pressure turns, just without
escalating wording; not filtered out here, flagged for `sycophancy_bank.py`/screening analysis to
account for if it matters.

Re-fetch/re-merge commands:

```bash
curl -s "https://raw.githubusercontent.com/JiseungHong/SYCON-Bench/master/false-presuppositions-setting/data/questions.txt" -o /tmp/sycon_questions.txt
curl -s "https://raw.githubusercontent.com/JiseungHong/SYCON-Bench/master/false-presuppositions-setting/data/presuppositions.txt" -o /tmp/sycon_presuppositions.txt
curl -s "https://raw.githubusercontent.com/JiseungHong/SYCON-Bench/master/false-presuppositions-setting/data/corrections.txt" -o /tmp/sycon_corrections.txt
curl -s "https://raw.githubusercontent.com/JiseungHong/SYCON-Bench/master/false-presuppositions-setting/data/push_back.csv" -o /tmp/sycon_pushback.csv
# then merge with csv.DictReader + zip() by line order into sycon_false_presuppositions.jsonl
# (see git history of this file for the exact one-off merge script used 2026-09-02)
```

`safemtdata_attack_600.json` contains multi-turn jailbreak attack query sequences (600 rows,
6 harm categories, each targeting an underlying harmful goal via 4-5 escalating in-context
questions about a real/fictional "actor" connected to that goal) used as **fixed, pre-generated
attacker input** for `docs/task/ADVERSARIAL_DEFENSE_TASK_FEASIBILITY.md`'s defense-screening task --
loaded by `attack_bank.py`. This is a published academic red-teaming benchmark (same one NBF-LLM
and other cited defense papers evaluate against); it is used here only to measure and defend an
open-weight model's own responses, not to develop new attacks.

Re-applying the two local patches to `download_file()` and `random_probes` is
manual after a re-fetch.

`mtbench_questions.jsonl` contains 80 benign multi-turn question pairs (10 per category, 8
categories: writing/roleplay/reasoning/math/coding/extraction/stem/humanities), each with a
fixed 2-turn `turns: [turn1, turn2]` follow-up. Used as **fixed, pre-generated benign session
content** for the Phase F helpfulness-cost check
(`docs/experiments/koopman_defense_pilot.md`) -- loaded by `benign_bank.py`, which chains 3
same-category entries into one 6-turn session so the Koopman-MPC controller has enough history
to act (mirrors how `attack_bank.py`/`safemtdata_attack_600.json` supplies fixed multi-turn
content for the adversarial-defense line).
