# WRITING_STATE

updated: 2026-09-22   by: fable session 01Fij5rw
last_commit: 940c077 (Experiments trim + MethodCropped figure); this commit archives stale plans
plan: paper/PLAN_finish_2026-09-22.md (steps 1-7 to finish the main text). Background project docs:
  `claude/page_budget_analysis_2026-09-21.md` (what to cut, classes A/B/C),
  `claude/experiments_trim_analysis_2026-09-21.md` (done), `claude/intro_and_related_work_plan_2026-09-21.md`
  (Intro done; Related Work plan for the user). Old plans: `_archive_2026-09-22_plans/`.
title: D41 2026-09-23 -> "KOMPAS: Koopman Dynamics Modeling and Black-Box Predictive Control of Multi-Turn LLM Reply Attributes"
state of sections:
  abstract.tex            WRITTEN 2026-09-23 (186 words, 8 sentences, user-reviewed; plan claude/abstract_conclusion_plan_2026-09-23.md); gates PASS
  01_introduction.tex     +1 phrase 2026-09-23 (P1: 'a safety score under jailbreak attempts', matches Abstract s1); gates PASS
  01_introduction.tex     rewritten 2026-09-21; trimmed 2026-09-22 to 1.0 pp (P3 removed, Method details out of P4,
                          contributions compressed; 786 words), gates PASS
  02_problem.tex          compressed 2026-09-21 (A1) and 2026-09-22 (inline labels, feasible-set paragraph); gates PASS
  03_method.tex           colleague's version (a1eeb79) + A4/A5 + round-2 trims + MethodCropped figure; 1063 words; gates PASS
  04_experiments + 04a-04d trimmed 2026-09-21 to trends + insight; gates PASS
  06_conclusion.tex       (was 05_) WRITTEN 2026-09-23 (167 words, 7 sentences, user-reviewed; two scope limitations, no point-estimate caveat per D40); gates PASS
  main.bib                CLEANED 2026-09-23: 30 entries, every author list complete (looked up on arxiv/HF/alphaxiv/mlanthology),
                          7 unknown* keys renamed (li2024instability, allbert2024personality, yang2025lfsteering, shi2026unisteer,
                          wang2026nautilus, ko2026attractor, luo2026spasm), genctrl duplicate dropped, 12 entries added (RE-Control,
                          TMPC, Akrout x2, Nosrati survey, Korda-Mezic, DMDc, Hankel-DMD, HAVOK, Lusch, Enyeart-Lin), printing
                          note fields removed. Journal volume/pages of the 2016-2018 Koopman papers from memory: verify.
                          Open: wang2026tmpc is text-level MPC per its abstract, conflicting with Table 2's white-box label (TODO in 05).
  05_related_work.tex     (was 06_) DRAFT 2026-09-23 (273 words, four categories A-D; all gates PASS except gate 5 = D7 no-\cite,
                          which fires by design here; missing bib keys listed as % CITE-TODO(bib) with arXiv links, user adds)
  2026-09-23 page-budget pass (claude/related_work_draft_and_page_levers_2026-09-23.md): Intro P2 compressed
                          18 -> 10 lines (one paragraph, one \uline gap sentence, pointer to Sec. 6); five displayed
                          equations inlined (eq:interaction, eq:control_objective, eq:memory_state, eq:koopman_fit,
                          eq:selection_bound; labels kept for the contract lattice); A6 now says "the selection bound of
                          Section 3.4". Body ends p.10 line 489 (budget 486): 3-4 lines over before the missing
                          citations are added (~+2). Remaining levers: RW drop 2 sentences (-2.5), \vspace on headings.
  appendix A1, A6, A7     written; A3 carries moved setup details + two TODO(data) tables; A2, A4, A5 TODO
page budget (ICLR 2027: 9 pp main text at submission):
  body ends p.9 line ~450 (Conclusion header at 447; ~8.3 pp incl. the three TODO stubs). Round 2 done
  2026-09-22: Intro to 1.0 pp; Method overview / lifting / 3.2 tail / 3.4 / Fig. 1 caption compressed and
  the four colleague >30-word sentences split; Problem subsections -> inline bold labels; 04a metric
  paragraph and 04b setup sentences compressed; Table 1 no longer stretched to \textwidth
  (build_tables.py threshold ncol>=10); Figs. 2-4 at 0.92\linewidth. Contract: eq labels reconciled
  (eq:koopman_fit, eq:predictive_control; eq_control_objective added). All section gates PASS.
  Budget: 9 pp = 486 lines; Abstract (~9) + Conclusion (~7 net) + Related Work at 0.4 pp (~21 net)
  land at ~487. ZERO margin: write Conclusion <=120 words, RW four categories x two sentences, and
  tune float/heading spacing with \vspace only if 1-3 lines remain.
gates: mech_audit.sh per file (see per-section notes above); latexmk exit 0, 12 pages, 0 undefined refs
open_questions_for_user:
  (1) colleague's constraint-satisfaction metric definition still missing; 04a and A3 carry TODO(data)
  (2) contract.yaml equation_lattice does not list eq:control_objective / eq:koopman_fit /
      eq:predictive_control (labels introduced by the colleague's method commit); reconcile
  (3) the letter C carries three objects (readout row, command range, candidate set); not renamed
  (4) whether to add a half-clause on the equal-cost boundary (A4) to Intro contribution 3
  (5) RESOLVED 2026-09-23: main.tex order is now Related Work -> Conclusion -> bibliography -> appendix; files renamed 05_related_work / 06_conclusion

---

# Session log (newest last)

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
| Fig. 2 | figs/fig_memory.pdf (a,b) | P1 memory depth, matched window (A1 matched) | c10 |
| Fig. 2 | figs/fig_memory.pdf (c) | P2 item intercept decomposition (M2); merged 2026-09-20, old fig_memory_depth/fig_item_effect removed 2026-09-22 | c11 |

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

## Introduction rewritten (2026-09-21, fable session)

Plan: `claude/intro_and_related_work_plan_2026-09-21.md` (project doc). `sections/01_introduction.tex`
rewritten in place (CP1 text backed up outside the repo); five paragraphs, 1077 words, spans p.1
line 17 to p.2 line 90 (about 1.37 pages with the abstract still TODO). What changed against CP1:

- P1: phenomenon recast from "drift" to carry-over / inertia, matching 04b + 04b_ablation (the
  memory gain is mostly a per-item level). Overshoot example and the untested `prior_4` driver removed
  (nc_9). One cited sentence on turn-level linear relaxation (Drift No More) motivates linear modeling.
- P2: now a prior-work paragraph with three bold categories (latent-space controllers /
  prompt-level remedies / control-theoretic views), each closed by a `\uline{}` gap; the three gaps are
  access, statefulness, and model. "Every controller" and "overshoots or undershoots" removed (A4
  ties the equal-cost schedule). `% CITE:` markers only (D7); two CITE-TODO(bib) notes for keys not in
  main.bib (korda2018koopman, akrout2026distinguishability, dmd2026safety).
- P3: interaction clock + verifier reading as the only observable, compressed; names the attribute
  types used (word count, sentence count, readability level, pass flag).
- P4: "three requirements -> unique structure" argument dropped (contradicted by LSTM win on
  constraint and by the lifting result). Now mirrors 3.1's Why: three-part intuition, window as
  dictionary, protocol gives the command a coordinate, precedent (Korda-Mezic) -> fit -> affine LS fit,
  explicit ARX admission, three things the operator view adds, lifting gloss (D33), controller loop,
  zero extra queries, replanning, key-insight sentence (c_t != r while the attribute is moving),
  bound in one sentence, KOMPAS revealed once.
- P5: contribution 1 = modeling from verifier readings alone with RQ1a + RQ1c evidence; 2 = RQ1b;
  3 = controller + selection bound + RQ2 numbers (14.5/14.8, 5.2/5.7, all 22 cells), D12 caveat kept.

Gates: `mech_audit.sh` PASS (sentence <=30 words / <=2 turns, equation, em-dash, adverb, cite,
terminology). `latexmk` exit 0, 13 pages (was 12); PDF at `build/main_introRewrite_2026-09-21.pdf`.
Not committed. Open for the user: whether to add a half-clause scope on the equal-cost boundary (A4)
to contribution 3; Related Work (`06_related_work.tex`) still TODO and is the user's to write.
Follow-up (same session): P1 re-grounded on the tables' attributes (word count, sentence count,
readability level, sentiment polarity, formatting constraint); persona-drift / attractor-state /
Drift-No-More-relaxation sentences removed from P1 (no persona line in the paper). P2 item 2 now says
"multi-turn instability", cites li2024instability + dongre2025driftnomore only. P3 gained one sentence
on verifier types (counting programs vs rater models). 1078 words, still ~1.35 pages; gates PASS.

## Page budget, class A (lossless) executed (2026-09-21, fable session)

Analysis: `claude/page_budget_analysis_2026-09-21.md` (project doc). ICLR 2027 main text <= 9 pages at
submission (10 at rebuttal/camera-ready). Before: body ended p.11 line 566 (~10.5 pp with Abstract,
Conclusion, Related Work still TODO). After A1-A5: body ends p.10 line 526 (~9.7 pp incl. the three
TODO stubs), i.e. 40 lines / ~0.74 page saved. PDF: `build/main_pageBudgetA_2026-09-21.pdf`.

- A1 `02_problem.tex` 564 -> 416 words: one lead sentence replaces the two per-subsection objective
  sentences; observability paragraph (D32) cut to one sentence (Intro P3 carries it); r-vs-c_t paragraph
  cut to two sentences; eq:modeling_objective inlined (A4). Eqs (1)(3) unchanged.
- A2 `04d_analysis.tex` 355 -> 299 words: bound + cost merged into one Findings paragraph; the
  command-channel scope moved to NEW `appendix/A7_command_channel.tex` (included in main.tex) with a
  two-sentence pointer in 4.6. The 38-word Question sentence (pre-existing violation) fixed.
- A3 `04a_setup.tex` 627 -> 498 words: per-task window/horizon ranges, row counts, trajectory counts,
  bootstrap resamples, unused-column note, second-backbone note and the reference-frame remark moved to
  `appendix/A3_window.tex` (was a TODO stub; now carries them plus two TODO(data) tables). eq:skill
  inlined (A4). The 31-word Joint-2 sentence (pre-existing) split.
- A4/A5 `03_method.tex` 1304 -> 1166 words: closed-form rollout inlined (eq:closed_form_rollout label
  gone; nothing referenced it); the two execution/replanning paragraphs of 3.3 merged; the duplicate
  learned-lifting paragraph of 3.2 removed; the 34-word ARX sentence of 3.1 split; one "directly" adverb
  removed. Eqs (4)(5)(6)(8)(9) unchanged.

Gates: 01/02/04a/04d/A3/A7 sentence gate PASS. `03_method.tex` still carries 4 pre-existing
sentence-length violations (lines 8, 11, ~205, ~221: colleague's overview and 3.4 text) and 3
pre-existing adverb hits ("directly" x2, "approximately"); none introduced here. equation_gate
reports eq:control_objective / eq:koopman_fit / eq:predictive_control as not in contract.yaml
equation_lattice: pre-existing label drift from the colleague's method commit, to be reconciled in
contract.yaml. latexmk exit 0, 0 undefined references, 12 pages total.

Remaining gap: with Abstract (~0.2), Conclusion (~0.25) and Related Work (~0.5) still to write, the
projected body is ~10.5 pp, i.e. ~1.5 pp over. Next: class B (float heights) and the class-C decisions.
Not committed.

## Experiments trim executed (2026-09-21, fable session)

Plan: `claude/experiments_trim_analysis_2026-09-21.md` (project doc). Rule applied: a number stays in
prose only if it is a claim headline, is not readable from the table/figure, or is a caveat; reading
rules go back to captions; collection properties go to the appendix. No table or figure touched.

| file | words | what moved |
|---|---|---|
| 04_experiments (lead) | 176 -> 116 | sub-question glosses cut to noun phrases |
| 04b_prediction | 338 -> 261 | column lists removed; CEFR +0.599 now via Fig. 4(c); LSTM two sentences merged; lifting driver moved to 4.4 |
| 04b_ablation | 541 -> 431 | filled/hollow rule and panel-(c) definitions back in the Fig. 2 caption; per-task depth values, remainder values, the "Joint-2 hollow" sentence and the duplicate closing sentence removed; +0.603 -> +0.109 caliber contrast moved to App. A3 |
| 04b_mechanism | 646 -> 471 | five reading-rule sentences back in the Fig. 3 / Fig. 4 captions (reverses eea9f36's move); Fig. 3 separations and Fig. 4(c) costs stated as trends; command-channel collection counts (2268/2268, 240/240, error span) moved to App. A7; uline shortened |
| 04c_control | 266 -> 219 | smallest-margin sentence and the unsupported "hardest to satisfy in one shot" sentence removed; the 22-cell count now lives only in the uline |

Evidence ids kept as line-end comments wherever a number left the prose. Gates: all touched files
PASS (02_problem still reports the pre-existing eq:control_objective contract-label drift).
Body now ends p.10 line ~503 (was 526 after class A, 566 before): -23 lines this pass, -63 in total.
Body ~9.3 pp incl. the Abstract / Conclusion / Related Work stubs; projected ~10.2 pp once they are
written, so ~1.2 pp still over the 9-page limit. Next levers: class B (float heights, ~0.25),
Related Work at 0.5 pp, Intro back to ~1.0 pp (~0.3), and the 4 pre-existing long sentences in
03_method. PDF: `build/main_expTrim_2026-09-21.pdf`. Not committed.

## Consistency audit, zero-cost fixes applied (2026-09-22)

Audit: `claude/fulltext_consistency_audit_2026-09-22.md`. Applied (no page cost): 5.2 -> 5.3 vs RE-Control on
Qwen3-4B (matches Table 2's rounded Avg column; unrounded 5.25 kept in comments) in Intro and 4.5; 4.3
Setup "which" misattribution fixed; 4.4 Question now "two mechanisms + one collection property"; Method
overview "first three subsections ... 3.4 gives the guarantee"; 4.6 bound sentence restated with delta_J's
actual meaning; "four self-built behavioral tasks" -> "four additional self-built tasks (App. A4)";
co-author / colleague / the authors wording in 4.1 neutralized (metric definition left as TODO(data)
comment only); Eq.~\eqref -> Eq.~(\ref) in 3.4 and App. A6 ("Eq. equation 7" gone); V called "verifier"
throughout 03_method and the Fig. 1 caption; "measurement(s)" -> "reading(s)" in 03_method; "linear
controlled regression" -> "delay regression with an exogenous input"; "encoder and decoder ... identity
map" -> "identity lifting"; intervals scope "4.2-4.4"; RQ lead now names 4.4; 2.2 r-vs-c_t paragraph
shortened (the c_t != r insight now appears in Intro, 3.3, 4.5 only); 4.1's App. B sentence corrected
(A2 = the four small-sample columns incl. sentiment and defense again; build_tables APPENDIX_COLUMNS
was right, the prose was wrong).
NOT done (need facts or the figure author): #1 vector attributes in Sec. 2 (check Joint task implementation),
#8 how each attribute is scored (program vs rater), #12 Fig. 1 selection-criteria box and typos
("Sliding Widow", "CLOSE LOOP"), App. B/D/E content, App. E unreferenced.
All section gates PASS. latexmk exit 0, 11 pages, 0 undefined refs; body ends p.9 line ~449.
PDF: `build/main_consistency_2026-09-22.pdf`. Not committed.
P1 items closed the same day: Table 1 caption "Free-running" -> "Multi-step prediction skill (forecasts from
step t only, no reading fed back within the horizon)" via build_tables.py (numbers identical); 4.2 driver
softened to "a level and, to a lesser degree, a trend" to agree with 4.3/4.4. Body still ends p.9 line ~449.
