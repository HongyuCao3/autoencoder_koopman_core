# Vendored resources

These files are vendored copies of third-party data used by the persona-drift
probe scoring functions, so that a compute node without internet access can
still import `hundred_system_prompts.py` and score probe responses.

| File | Source | Upstream commit/version | License |
|---|---|---|---|
| `hundred_system_prompts.py` | `Naomibas/llm-system-prompts-benchmark` on Hugging Face | sha `0e5379e1465827c1d00d7c4102b793e5be269e77` (fetched 2026-08-27) | Apache-2.0 |
| `word_lists/google-10000-english-usa.txt` | `first20hours/google-10000-english`, `google-10000-english-usa.txt` | `main` branch (fetched 2026-08-27; upstream `hundred_system_prompts.py` still pointed at the old `master` branch, which 404s) | see upstream repo |
| `word_lists/one-syllable-sorted-by-prevalence.txt` | `gautesolheim/25000-syllabified-words-list`, `1-syllable-sorted-by-frequency.txt` | `main` branch (fetched 2026-08-27; file was renamed and the `master` branch removed upstream since `hundred_system_prompts.py` was written) | see `LICENSE.md` in that repo |

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
```

Re-applying the two local patches to `download_file()` and `random_probes` is
manual after a re-fetch.
