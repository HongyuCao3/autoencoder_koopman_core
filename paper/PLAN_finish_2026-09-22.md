# PLAN — resolve the remaining inconsistencies and finish the main text

updated: 2026-09-22 (fable session 01Fij5rw). Read after the status block of `WRITING_STATE.md`.
Rulings live in `DECISIONS.md`; the audit this plan derives from is project doc
`claude/fulltext_consistency_audit_2026-09-22.md` (everything it marks done is in the tree at this commit).
Budget context: body ends p.9 line ~449; ICLR 2027 limit is 9 pp = 486 lines; Abstract + Conclusion +
Related Work still to write (~37 lines). Every step below states its line cost.

Hard rules for every step: sentences <= 30 words and <= 2 logical turns (D8); numbers only from
`evidence/*.yaml` with the id in a line-end comment; no `\cite` (D7, `% CITE:` markers); no
test-time / intervention / alignment (D29); gate with `bash tools/mech_audit.sh sections/<file>.tex`;
never edit `tables/*.tex` by hand (regenerate with `python3 tools/build_tables.py`).

---

## Step 1 — Facts to look up (no writing yet)

| id | question | where to look | feeds |
|---|---|---|---|
| F1 | How are Joint-2 / Joint-3 (`vector_count_stage{1,2}_t10`) modeled: one vector reading $y_t\in\mathbb{R}^{2\mid 3}$ stacked in the window, or one scalar model per coordinate? | `src/koopman_ae` (dataset/readout code for `vector_count_*`), `configs/dataset/vector_count_stage*.yaml`, `docs/NAMING.md` | Step 2a |
| F2 | For each of the seven Table 1 tasks, is the reading produced by a program (counts, constraint checker) or by a rater model? Which model rates sentiment, defense, CEFR? | `DATASETS.md`, `docs/LEDGER.md`; `evidence/numbers.yaml` (defense: `judge_kind=self`, n_main_136-151; CEFR seed caveat n_mech_068-070); 04a already says Qwen3-4B-Instruct-2507 "both produces and scores" | Step 2b |
| F3 | Colleague's per-subtask definition of a satisfied instruction (the control metric) | colleague; blocks D30's feasible-set bridge as well | Step 2c |
| F4 | The four extra self-built tasks of the equal-cost comparison, the fixed schedules compared, per-task numbers | `evidence/numbers.yaml`; prose in `_archive_2026-09-18_negative_paper/sections/` | Step 4 (A4) |

## Step 2 — Fix the three remaining P0 text inconsistencies (after Step 1)

| id | file | change | lines |
|---|---|---|---|
| 2a | `02_problem.tex`, Modeling paragraph | Replace "We track one such attribute at a time" by one sentence stating the vector case per F1, e.g. "The formulation is written for a scalar attribute; a vector attribute stacks its coordinates in the window and $C$ takes one row per coordinate." If F1 says per-coordinate models, say that instead. Check that §3.1's $C=[1,0,\dots,0]$ sentence still reads correctly (add "for a scalar attribute") | +1 |
| 2b | `04a_setup.tex`, task paragraph | One sentence naming the two verifier kinds and which tasks use which (per F2). This restores what §3.1 promises ("the definition of $V$ is given in the experimental setup") | +1 |
| 2c | `04a_setup.tex`, metric paragraph | Write the satisfaction definition (per F3); delete the `% TODO(data)` comment | +1 |

Compile, gate, record the new body end line in `WRITING_STATE.md`.

## Step 3 — Figure 1 (figure author; text unchanged)

| id | change |
|---|---|
| 3a | Bottom-right selection box: replace the three criteria (Min final state / Min average / Min final control) by the single cost of Eq. (6), e.g. "$\min_{c}\widehat{J}_t(c)$ over the predicted trajectory" |
| 3b | Typos: "Sliding Widow" → "Sliding Window"; "ONLINE KOOPMAN CLOSE LOOP" → "CLOSED LOOP" |
| 3c | Optional: rename "Control 1..N" / "Control state" to "command $c$" / "state $s_t$" to match the text |

Re-export to `figs/MethodCropped.pdf` (same aspect ratio ~3.2:1 so the page layout does not move).

## Step 4 — Appendices that the main text already points to

| id | file | content | note |
|---|---|---|---|
| 4a | `appendix/A4_equal_cost.tex` | the four tasks (F4), the equal-cost fixed-schedule protocol, per-task numbers; **also** the control-line run protocol, because §4.1 now says "produced in a separate pipeline whose protocol Appendix D records" | if the protocol does not belong in A4, change the §4.1 pointer instead |
| 4b | `appendix/A2_small_columns.tex` | `\input{tables/tab_prediction_small.tex}` (already generated; columns = sentiment, defense, average word length, GSM8K-sharded, by D18) + one sentence on the sample-size caveat | keep sentiment/defense duplication: intentional |
| 4c | `appendix/A3_window.tex` | fill the two `TODO(data)` tables (per-task window/horizon/rows/trajectories; matched-window numbers behind Fig. 2) | numbers from `evidence/numbers.yaml`, `figs/_fig_report.json` |
| 4d | `appendix/A5_onestep.tex` | either add a one-clause pointer from §4.1's skill paragraph ("one-step errors are in Appendix E") or delete the appendix and its `\input` | currently unreferenced |

## Step 5 — Write the three missing main-text pieces (budget-bound)

Order: Related Work → Conclusion → Abstract (Abstract is extracted from the others).

| id | file | budget | must contain / must match |
|---|---|---|---|
| 5a | `06_related_work.tex` | ~0.4 pp: four `\textbf{Category.}` labels × (one description + one gap sentence), third person, no first-person plural, no closer | plan in project doc `claude/intro_and_related_work_plan_2026-09-21.md` §四 (A drift measurement / B latent-space control / C control-theoretic views / D Koopman predictors with inputs). Decide position: `main.tex` has Conclusion → Related Work; the skill wants Related Work before Conclusion |
| 5b | `05_conclusion.tex` | ≤120 words, takeaway-first 6-move | limitations list: benchmark numbers are point estimates without intervals; closed loop ties the equal-cost schedule on the four A4 tasks; TMPC / RE-Control are reproductions; single backbone on the modeling line |
| 5c | `abstract.tex` | ≤150 words, no setup details | numbers identical to Intro contribution 3: 14.5 / 14.8 over Base, **5.3** / 5.7 over RE-Control, point estimates; KOMPAS revealed once |

After 5c: compile; if the body exceeds 486 lines by 1–3 lines, tune `\vspace` around floats/headings; if by more, shorten 5a first, then 5b.

## Step 6 — References and contract

| id | change |
|---|---|
| 6a | `main.bib`: add `korda2018koopman`, `kong2024recontrol`, `wang2026tmpc`, `akrout2026distinguishability`, `dmd2026safety`; fix the `unknown*` keys; merge the two GenCtrl entries. Then turn every `% CITE:` marker into `\cite` (D7 lifted) and run `bash tools/mech_audit.sh` with the cite check disabled |
| 6b | `contract.yaml`: c6 notes should carry both 5.25 (unrounded) and 5.3 (Table-rounded, as printed); confirm `claim_ledger` numbers against Intro/4.2/4.5 once more |

## Step 7 — Final read

- Re-run the consistency pass of `claude/fulltext_consistency_audit_2026-09-22.md` §P1 on the new text (verifier / reading / delay regression / identity lifting vocabulary; no "colleague", "co-author", "TODO" in the PDF).
- `pdftotext main.pdf - | grep -n "TODO\|colleague\|co-author\|equation [0-9]"` must return nothing.
- Update the status block of `WRITING_STATE.md`; commit.

## Open user decisions (do not resolve silently)

1. Related Work before or after Conclusion (`main.tex` order).
2. Whether Intro contribution 3 gets a half-clause on the equal-cost boundary (A4).
3. The letter $C$ for three objects (readout row $C$, command range $\mathcal{C}$, candidate set $\mathcal{C}_t$): rename or leave.

## How to resume on another machine

1. `git pull`; read the status block of `WRITING_STATE.md`, then this file; start at the first unchecked step.
2. Build: `cd paper && latexmk -pdf -interaction=nonstopmode main.tex` (TeX lacks `algorithmic.sty`; `main.tex` already works around it).
3. Windows checkout: `git config core.autocrlf true` first (see project doc `claude/repo_git_hygiene_notes_2026-09-18.md`).
