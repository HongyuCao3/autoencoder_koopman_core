# WRITING_STATE

updated: 2026-09-19   by: opus session 01TAse1G
plan: WRITING_PLAN_2026-09-19.md (v2.1)
phase: R2 done   checkpoint: CP2-redo   status: awaiting_user_review
last_commit: (this commit)
artifacts_ready: contract.yaml (D20 signed; v2.1 revision applied this session, D25-D35),
  semantic.md §Introduction (approved CP1), §Method (rebuilt for CP2-redo), §Experiments (draft,
  relabel partly done), sections/01_introduction.tex (contribution 2 split), sections/02_problem.tex
  (D30 bridge with TODO(data), D32 observability, forward ref to sec:model),
  sections/02_method.tex (rewritten), sections/appendix/A6_selection_bound.tex (new),
  sections/appendix/A1_estimator.tex (opening rewritten)
gates: sentence=pass equation=pass (10 displayed equations in Method) term=pass mech=pass
  compile=pass (build/main_CP2redo_2026-09-19.pdf, 10 pages; Method spans pages 3-5)
cold_review: audit/method_T3_redo_2026-09-19.yaml (0 blocker / 2 major / 5 minor; 5 fixed,
  1 deferred to the user, 1 rejected with a reason)
user_verdict_on_previous: CP1 approved; CP2 superseded by this rewrite
next_action: user reviews §3 (and the four subsection titles); then Phase 3 relabel pass and 03a-03d
open_questions_for_user:
  (1) the colleague's constraint-satisfaction metric definition is still missing; §2.2's bridge and
      4.1 P5 carry TODO(data) until it arrives
  (2) D31, Figure 1 module C, deferred by the user this session. The figure is still the colleague's
      four-module control pipeline and the image itself contains the words "Test Time", which the
      term gate cannot see because it is a PNG. Phase R3 has to redraw or relabel it
  (3) cold review F3: the letter C carries three objects (readout row, command range, candidate set).
      Renaming touches the colleague's notation, the appendices and the figure, so it was not done
      unilaterally

## What CP2-redo changed

Method is a rewrite, not the CP2 reshuffle. Old outline: problem / state / operator / selection /
bound. New outline, cut by component (D25):

| new | content | equations |
|---|---|---|
| 3.1 `sec:model` A Koopman Dynamics Model of Prompt-Response Interaction | three-part intuition, the state window as a paragraph, the operator and the sentence tying delay coordinates to it as a dictionary, the protocol, the affine model in s-space, the lifting gloss (D33), one Why paragraph arguing modeling capacity and answering the ARX objection | memory_state, koopman_operator, prompt_protocol, koopman_model |
| 3.2 `sec:learning` Learning the Model from Interaction Data | transition tuples moved back from A1, the least-squares fit, coverage, a pointer to A1 for the learned lifting. No Why paragraph | **identification (new)** |
| 3.3 `sec:controller` Predictive Instruction Selection with Replanning | Why paragraph (black-box scoring), rollout, closed form, selection, Algorithm 1, replanning scope | rollout, closed_form_rollout, selection |
| 3.4 `sec:guarantee` What Prediction Accuracy Buys | Proposition 1 in s-space, one-sentence proof idea, boundary | prediction_bound, selection_bound |

Everything with a Lipschitz constant, a spectral-norm bound or a lifted coordinate moved to the new
appendix A6. The main text's equations use only s-space symbols; `E_phi` appears once, inline, in
the lifting gloss.

## Tool changes this session

- `mech_audit.sh`: check 8, the D29 terminology gate (test-time / intervention / alignment, with a
  `% GATE-EXEMPT: term` escape and an exemption for `\begin{aligned}`); check 9, the D33 lifting-gloss
  screen (first paragraph naming a lifting must contain "identity", an analogy, and at most one
  inline symbol).
- `equation_gate.py`: skips the body of an `algorithmic` block, and treats a proposition's opening
  line as transparent so the intuition sentence above the environment counts.
- `sentence_gate.py`: `algorithm` / `algorithmic` added to the stripped environments; a theorem
  environment's optional title is treated as a label, not a sentence.
- `main.tex`: this TeX installation has no `algorithm.sty`, so the float is declared with
  `float.sty` and Algorithm 1's body is a numbered list. `amsthm` supplies the proposition
  environment. Appendix A6 is included.

## New decisions

D25-D35 are in `DECISIONS.md`. D31 (Figure 1 module C) is recorded as deferred, not decided. D35
records that the D22 duplicate the plan warned about does not exist in the table, so no renumbering
was done.

## Phase 3 note (unchanged from CP3 start)

`semantic.md` §Experiments still carries its four subsections in draft. The relabel pass is partly
done: `03a_setup.tex` and `03d_analysis.tex` now point at `sec:model`, `sec:controller` and
`sec:guarantee`, and 4.4 points at Proposition 1. The rest of §5.5's relabel checklist runs before
03a-03d are generated.

## Side experiments P1 / P2 added (2026-09-19, fable session)

Plan: `claude/side_experiments_plan_2026-09-19.md` (project doc). New subsection
`sections/04b_ablation.tex` = printed 4.3 "What the Memory State Carries", inserted between
04b_prediction and 04c_control in `04_experiments.tex`. Two figures, one per experiment, both
generated by `tools/build_figs.py` from `evidence/numbers.yaml` (ids per mark in
`figs/_fig_report.json`; nothing typed by hand, same rule as build_tables.py):

| fig | file | experiment | claim |
|---|---|---|---|
| Fig. 2 | figs/fig_memory_depth.pdf | P1 memory depth, matched window (A1 matched) | c10 |
| Fig. 3 | figs/fig_item_effect.pdf | P2 item intercept decomposition (M2) | c11 |

contract.yaml: claims c10, c11 (modeling line, after c4); mechanism m6 (per-item level). Both
claims cite caveated ids (n_abl_151 two-cell design; n_mech_068-070 CEFR synthetic seed) and carry
the caveat in `notes`; the prose states both. `check_claim_evidence.py` passes for c10/c11 (its
n_bench_* misses for c5-c7 are pre-existing: those ids live in numbers_benchmark.yaml).
gates: mech_audit on 04b_ablation.tex = PASS (sentence, equation, em-dash, adverb, cite, term).
compile: build/main_sideP1P2_2026-09-19.pdf, 11 pages (was 10); the new subsection spans pages 7-8.
KNOWN, NOT MINE: `latexmk` exits 12 because colleague commit a1eeb79 ("Update Method") reintroduced
`\begin{algorithmic}` in 03_method.tex and this TeX installation has no algorithmic.sty (main.tex
comment lines 22-24). The PDF still builds; Algorithm 1 renders broken until that is resolved.
Not committed to git (user did not ask).
