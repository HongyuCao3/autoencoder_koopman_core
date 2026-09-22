<!-- STALE (2026-09-22): this Tier-2 outline is no longer maintained. §Introduction was superseded by the
2026-09-21 rewrite of 01_introduction.tex; §Method by the colleague's 03_method.tex (a1eeb79) and the
class-A trims; §Experiments by the 2026-09-21 trim. Kept for provenance of claim ids only. -->
# Tier 2 — semantic outline

## §Introduction

<!-- status: approved CP1 (Tier-2 cold review intro_T2_2026-09-19.yaml: 0 blocker, 2 major, 2 minor; all four triaged 2026-09-19) -->

P1 (motivation): opens on multi-turn interaction turning a frozen model's reply attributes into a trajectory that drifts turn over turn; grounds the drift in inertia via [prior_1] (attributes carrying over from recent turns rather than resampling fresh each turn); hooks with failure instance [fx_1] verbatim — "a rewrite pushed past the level it was aiming for"; offers a candidate cause for the drift, an instruction issued once competing against everything said after it, sourced to [prior_4]; carries [prior_4] in a hedged verb inside the same clause (one candidate account among others), never as a measured result, and never as a separate disclaimer sentence — [nc_9] is audit-only under D16 and must not surface as prose; a `% CITE:` placeholder line carries its borrowed-authority set (li2024instability, driftnomore2025, attractorstates2025); no formal object introduced, no claim_ledger id attached to the candidate-cause clause

P2 (gap): three gaps, each a prior-work label paired with a However-clause (Rule 18 pairing); Gap 1, wrong interface — label: existing controllers acting inside the network, white-box access needed at every token; However-clause: the deployable lever being one instruction per turn against a black-box API, grounded in [prior_2]; Gap 2, stateless and open loop — label: a system prompt written once, a reminder on a fixed schedule, a per-turn classifier template; However-clause: none predicting what the next instruction will do, overshoot or undershoot as the result; Gap 3, no usable dynamics model at the interaction layer — label: a control-theoretic framing of prompting already existing; However-clause: no model computable from one instruction to a predicted next-turn attribute accompanying it; no formal object; no claim_ledger id required for gaps 2–3

P3 (intuition): grounds the interaction-level time step in [prior_3] — the control objective a property of a whole reply, the verifier running only on a finished reply; contrasts per-token intervention, needing white-box access, against per-interaction intervention, needing only the ability to change a prompt; carries the room-temperature analogy — controlling a room's temperature needing no model of each air molecule, controlling a reply's attribute needing no model of each token; zero formulas, zero symbols, zero numbers throughout this paragraph (hard constraint); precedes every formal-object bullet in this section

P4 (bridge): compresses five predictor requirements into a three-clause triad (Rule 3 parallel triad), previewing [choice_2] ahead of its formal introduction in §Method — admitting nonlinearity in how a frozen model responds to an instruction; keeping the latent step linear, so a horizon of candidates becomes a matrix power rather than additional generations; keeping the command an explicit, readable channel rather than a target for text regression; notes LSTM meeting clause one and missing the other two; notes a per-turn Markov rule meeting clause two and missing the other two; reveals **KOMPAS** bold inline, exactly once, as the minimal structure meeting all three clauses at once; no equation, no symbol-table entry introduced here, formal treatment deferred to §Method

P5 (contributions): numbered exactly three, one per [choice_1, choice_2, choice_3] in that order; joint training objective excluded per D19 (moved to appendix A1), not listed as a fourth item; item 1 carries [choice_1]'s intro_bullet_keyword — a finite-memory state over observable attributes, no hidden-state access; backed by [c1] (history beyond the current turn needed on five of seven tracked attributes, paired-difference CIs excluding zero there); item 2 carries [choice_2]'s intro_bullet_keyword — a command-conditioned operator whose linear, delay-embedded special case is the instance reported; backed by [c5] (predictive control raising constraint satisfaction over the same frozen model, a 14.5-point average gain on Qwen3-4B and 14.8 on Qwen3-8B); item 3 carries [choice_3]'s intro_bullet_keyword — model-predictive selection of the next instruction, black-box, no extra generation per candidate; backed by [c6] (the same controller exceeding two white-box latent controllers, a 5.2-point average margin on Qwen3-4B and 5.7 on Qwen3-8B) and by [c8] (evaluating a candidate costing one latent propagation and a decode, no added query to the target model); exactly three items total, no fourth; ONE shared trailing clause after the list, not repeated per item, satisfies D12 by naming every control number above as a point estimate on three public instruction-following benchmarks (IFBench, IFEval, COLLIE; see meta.benchmarks) with no interval attached; budget discipline (Tier-2 cold review f3): each item is keyword plus ONE headline number, the per-model split and the margin over white-box controllers move to §Experiments, so the three items read as parallel triples rather than a wall; if P5 still exceeds its share at compile time, P2 merges two of its three gap labels before anything in P5 is cut

## §Method

<!-- status: rebuilt 2026-09-19 for CP2-redo (D25). Supersedes the CP2 block, which followed the
     colleague's controller-paper outline (state / operator / learning / test-time control / bound).
     Subsections are cut BY COMPONENT (model / learning / controller / guarantee), not one per
     design choice: choice_1 and choice_2 both live in 3.1 and the state window is a paragraph.
     Why paragraphs: at most one per subsection (D32); 3.2 and 3.4 carry none. -->

Opening paragraph (<= 4 sentences, before 3.1, modeling before control): one prompt-response
exchange is one step of a dynamical system; on delay coordinates of the verifier's reading we
identify a command-conditioned Koopman dynamics model; the model then scores candidate
instructions before any of them is sent; the LLM's parameters never change and the model is fit
offline. Figure~1 referenced as left column modeling, right column control. BANNED opening: "We
formulate ... as feedback control" (D25). No numbers, no baseline names (Rule 21).

### 3.1 A Koopman Dynamics Model of Prompt-Response Interaction  (label sec:model, ~0.85 page)

3.1 P1 (intuition, no symbols): the next reading is what the recent history carries forward, plus
what the instruction pushes, plus a constant offset; that three-part reading of one step is the
whole model in words, before any symbol appears.

3.1 P2 (anchor + Eq + gloss): CLAIM FIRST, then the example — a single reading is consistent with
two opposite trends, so a memoryless rule issues the same command where the next step differs
[m1]; [ex_1] verbatim as the illustration inside that claim, "the same sentence length can occur
while successive revisions are becoming longer or while they are becoming shorter"; must NOT open
with "For example" (CP2 defect). Introduces [eq_memory_state]; glosses [s_s, s_L, s_C]; states
[as_1]'s plain statement and its scope (an approximation to the information needed for
prediction, not a reconstruction of the dialogue state); marks [as_1] as tested in the prediction
experiments. The observability argument for windowing a verifier's reading is NOT here: it moved
to Problem Formulation 2.1 as an observability fact (D32). Home of [choice_1].

3.1 P3 (intuition + Eq + gloss): the population object before any approximation is the conditional
expectation of an observable one step ahead given state and command; introduces
[eq_koopman_operator]; glosses [s_Kop]; IMMEDIATELY AFTER the gloss, one sentence tying the two
halves together — the delay coordinates [s_s] are themselves the dictionary of observables, so the
affine model below is this operator's finite-dimensional approximation on that dictionary, which
is what makes the linear instance a Koopman model rather than a separate regression. Note the
operator is linear in the observable although the prompt-response dynamics need not be. % CITE:
arbabi2017hankel, korda2018mpc  (delay observables / lifted MPC; the Related Work phase resolves
these keys)

3.1 P4 (Eq + gloss): the command coordinate's provenance, compressed to one short paragraph — every
prompt is built by a fixed protocol from a scalar command, so the coordinate is a property of the
protocol rather than of the instruction's surface wording; introduces [eq_prompt_protocol]; glosses
[s_P, s_c, s_Cspace]; scope kept verbatim — the specified protocol only, not arbitrary text inputs
with an unspecified encoding. NOT a Why paragraph any more (D26 re-aimed 3.1's Why at modeling
capacity).

3.1 P5 (Eq + gloss): intuition — with the state and the command coordinate fixed, the model is the
plainest map that can carry all three parts of P1; introduces [eq_koopman_model] IN S-SPACE ONLY,
one-step affine transition plus command column plus bias, read out by the selector row; glosses
[s_K, s_B, s_b]; states that the shared transition and the affine command dependence are modeling
restrictions rather than consequences of the operator's linearity, instantiating [as_2] (now "affine
on the delay coordinates", tested by rq2b) and [nc_4]. No encoder, decoder, latent state or readout
symbol appears in this equation (D25, §5.3 notation list).

3.1 P6 (lifting gloss, D33): 4-6 sentences, no display math, at most the inline symbol $E_\phi$
once; written for an ML reviewer who has never read a Koopman paper; in order — (1) what a lifting
is: a change of coordinates, the same affine step written on learned features of the window instead
of on the window itself; (2) ONE analogy from [prior_5], the pendulum angle versus its sine and
cosine, OR feature choice in linear regression, exactly one of the two, not both; (3) this paper
reports the identity lifting, so the model is written directly on the readings, and the learned
variant is kept as a comparison row trained as in appendix A1; (4) why a learned lifting has little
to expose when the reading is a scalar and the window is short [m4], as a mechanism with NO number
(Rule 21). Every later mention of "lifting" in the paper points back here and does not re-explain.

3.1 P7 (Why paragraph, the only one in this subsection): **Why a Linear Operator on a Short Window
Can Carry Multi-Turn Dynamics** — the modeling argument for [choice_2]. It concedes the ARX
objection in the open (the reported instance is, numerically, a delay regression with an exogenous
input), then names what the operator view adds that the regression view does not: the operator acts
on observables, so the dictionary can be swapped and the learned lifting is the same model rather
than a different one; the multi-step forecast has a closed form instead of a simulation loop; and
the resulting predictor is the object standard predictive control already knows how to plan with.
Formula-free, number-free, symbol-free (Rule 22). Sentence shape must NOT be "X is standard, so the
open question is Y" — that shape is now used once at most in the whole section and 3.3 owns it.
Ends with the \uline{} answer sentence (Rule 18).

### 3.2 Learning the Model from Interaction Data  (label sec:learning, ~0.4 page, NO Why paragraph)

3.2 P1 (setup): construction of the transition tuples, moved here from appendix A1 — each window of
$L+1$ readings gives one state, sliding the window by one step gives the next tuple, and the command
label is the command applied before the successor observation; for trajectories recorded under a
fixed requested target, that target labels every step.

3.2 P2 (intuition + Eq + gloss): intuition — in the reported instance the only unknowns are the
transition, the command column and the bias, and all three enter the next-step prediction linearly,
so fitting them to recorded transitions is an ordinary least-squares problem with a closed-form
solution; introduces [eq_identification]; glosses the sum over overlapping windows; one sentence on
scale — the unknowns number in the tens, which is why tens to hundreds of recorded trajectories
suffice (a property of the parameterization, no dataset number here, Rule 21). Supports the
offline-fit half of [c8].

3.2 P3 (scope): coverage — accurate prediction on fixed-command training trajectories does not by
itself establish accuracy after command changes, so what the training trajectories cover in
command-history pairs sets where the model can be trusted at inference time (moved here from the
old 3.3 closing paragraph). Registers [nc_7] as audit-only: no term-removal ablation is claimed.

3.2 P4 (pointer): one sentence — the learned-lifting variant replaces the identity map with an
encoder and fits it jointly with the transition under a three-term objective, given in appendix A1;
points back to 3.1 P6 for what a lifting is, does not re-explain it.

### 3.3 Predictive Instruction Selection with Replanning  (label sec:controller, ~0.65 page)

3.3 P1 (Why paragraph, the only one here): **Why Scoring Happens Without Opening the Model** —
argues [choice_3], kept from CP2 in substance: planning on a linear predictor is standard predictive
control, so what has to be justified is scoring a candidate without opening the model that would
otherwise produce the reply; each candidate is compared only on the reply attributes it is predicted
to produce. Formula-free, number-free, symbol-free (Rule 22); ends \uline{a candidate is judged only
by the reply attributes it is predicted to produce, never by anything inside the model that would
produce them}; instantiates [prior_3]. Sentence shape differs from 3.1 P7's (D25 §5.3).

3.3 P2 (intuition + Eq + gloss): intuition — evaluating a candidate means holding its command fixed
over a lookahead and rolling the state forward, with no further generation; introduces [eq_rollout]
IN S-SPACE; glosses [s_Cset, s_H].

3.3 P3 (intuition + Eq + gloss): intuition — an affine transition with the command in one column
collapses an h-step lookahead into a matrix power [m2]; introduces [eq_closed_form_rollout]; no new
symbol; one sentence of consequence — evaluating a candidate costs a matrix product and a readout
and adds no query to the target model, supporting [c8] as a structural property rather than a
measurement.

3.3 P4 (intuition + Eq + gloss): intuition — scoring the whole predicted trajectory rather than the
immediate effect lets the controller issue a command different from the target while the attribute
is still moving [m3]; introduces [eq_selection]; glosses [s_w, s_Jhat].

3.3 P5 (Algorithm 1, D34): the receding-horizon loop as pseudocode — read the attribute of the last
reply; shift it into the window to form the state; for each candidate command, roll the state
forward over the horizon in closed form and score the predicted trajectory; take the argmin; build
the instruction from it with the protocol; send it, read the new attribute, repeat until the budget
is spent. No result numbers anywhere inside the environment (Rule 21). The caption names it as the
loop, not as a contribution.

3.3 P6 (scope): only the first step of the plan is executed, replanning starts from the actual
reading rather than from a predicted one, and the constant-command assumption lives only inside a
hypothetical rollout; candidate simulation adds no target-model query, while constructing candidates
and reading replies still counts against the interaction budget.

### 3.4 What Prediction Accuracy Buys  (label sec:guarantee, ~0.25 page, NO Why paragraph, D28)

3.4 P1 (intuition, one or two sentences): the more accurately the model predicts each candidate's
trajectory, the closer the selected command has to be to the one that would have been best in
hindsight; this sentence is the intuition the equation gate requires ahead of the proposition.

3.4 P2 (Proposition 1): stated in s-space inside a `proposition` environment; assumptions carried in
words (the one-step state residual and the readout residual stay below fixed levels over the
candidate rollouts, and the readout does not amplify small state differences without bound), with
[s_eps] named; conclusion in two lines, the h-step attribute error bound [eq_prediction_bound] and
the selection bound [eq_selection_bound], with [s_Delta, s_deltaJ] glossed; supports [c9],
strength_basis theoretical. The Lipschitz constant, the spectral-norm bound, the uniform range R and
the full assumption display stay in appendix A6 (D28).

3.4 P3 (proof idea + pointer): one sentence — the predicted and actual rollouts start from the same
state, so their gap accumulates only through the one-step residual, and applying the cost bound
twice around the argmin gives the factor two; full assumptions and proof in appendix A6.

3.4 P4 (scope, ends the section): the bound is about one held-command comparison; it does not
guarantee that the candidate set contains a command able to reach [s_r], and it does not guarantee
that the replanned closed loop converges [nc_5]; ends with the \uline{} answer sentence (Rule 18).

### Method-wide constraints (Tier 3 must satisfy all of them)

- Notation: main-text equations use only $x, p_\theta, \mathcal{H}_t, u_t, o_t, y_t, V, r, e_t, c_t,
  \mathcal{C}, \mathcal{C}_t, \mathcal{P}, s_t, L, C, K, B, b, H, h, w_h, \widehat J_t, c_t^\star,
  \Delta_h, \delta_J, \varepsilon_{\mathrm{dyn}}, \varepsilon_{\mathrm{rec}}$. $D_\psi, g_\psi,
  \xi_t, d_\xi, L_g, \kappa, R$ appear only in appendices A1 and A6; $E_\phi$ appears once inline, in
  3.1 P6.
- Terminology (D29): instruction for $u_t$, command for $c_t$, reading or attribute for $y_t$,
  inference-time rather than test-time. The words test-time, intervention and alignment do not occur.
- Why paragraphs: 3.1 P7 and 3.3 P1 only, with different sentence shapes, each ending in a \uline{}
  answer.
- Equations by home: 3.1 carries eq_memory_state, eq_koopman_operator, eq_prompt_protocol,
  eq_koopman_model; 3.2 carries eq_identification; 3.3 carries eq_rollout, eq_closed_form_rollout,
  eq_selection; 3.4 carries eq_prediction_bound and eq_selection_bound inside Proposition 1.
- Labels: sec:model, sec:learning, sec:controller, sec:guarantee. The old labels sec:memory_state,
  sec:koopman_operator, sec:control and sec:analysis are retired and every reference to them, in
  §2, §4 and the appendices, is updated in the same pass.

## §Experiments

<!-- status: draft CP3 -->

RQ-mapping note (stated once, applies to all four subsections below): `rq_id` values in contract.yaml (rq1, rq2a, rq2b, rq3) are internal identifiers and are never printed; the reading-order numbering used in the prose is printed RQ1 = contract rq2a (history, tested in 4.2), printed RQ2 = contract rq2b (lifting/sequence-model, tested in 4.2), printed RQ3 = contract rq1 (control, tested in 4.3), printed RQ4 = contract rq3 (bridge and cost, tested in 4.4); D17 places 4.1 Setup ahead of both lines, then the modeling line (4.2) ahead of the control line (4.3), then the bridge (4.4).

BUDGET note: 4.1 carries 5 setup paragraphs, 4.2 and 4.3 carry 4 each (Question+Setup, two Findings, one closing), 4.4 carries 6 (Question+Setup, bound, cost, two Cross-check scope paragraphs, one closing); 19 paragraphs total against the section's 2.5-3.0 page budget including both tables is tight, so 4.4's two Cross-check paragraphs are the first candidate to merge into one paragraph at Tier 3 if the section overflows.

### 4.1 Experimental Setup

4.1 P1 (setup): two task groups; group one is the self-built multi-turn tasks used for the modeling line, eleven total, seven carried in Table 1 (sentence length, items-s2, items-s1, character length, formality, constraint, CEFR) and four carried in an appendix table (sentiment, average word length, defense, GSM8K-sharded) [tab_prediction; `_build_report.json:prediction_main` and `:prediction_appendix` keys]; group two is the three public instruction-following benchmarks used for the control line, IFBench, IFEval and COLLIE, eleven subtasks total (WC/UWC/NS/NP, KF/WC/NS/NP, Word/Sent./Para.) [meta.benchmarks]; RECOVERED 2026-09-19 (Opus, from docs/LEDGER.md lines 201-202 and 549): the self-built trajectories are produced and judged by **Qwen3-4B-Instruct-2507**, the same agent model across the attribute-tracking and behavioral lines; docs/NAMING.md's `backbone2` row confirms the existing Table 1 is a single-backbone result and that the gemma second-backbone arm is a plan with no submitted GPU job, so it is NOT reported

4.1 P2 (setup): modeling-line baselines are the six rows of Table 1, Stateless (memoryless null), Last-turn (single-reading state), Memory w/o action (multi-turn state without the command term), LSTM (recurrent sequence baseline), KOMPAS (learned lifting), KOMPAS (linear, the reported instance) [tab_prediction; instantiates [choice_1] and [choice_2]'s linear special case]; RECOVERED 2026-09-19 (Opus, tabulated from evidence/numbers.yaml; every task has exactly one consistent setting, verified by grouping all 183 n_main_* rows): memory length and horizon per task are sentence_length lag 3 / H 6 / 252 scored rows, items-s1 lag 3 / H 6 / 576, items-s2 lag 3 / H 6 / 1620, character length lag 1 / H 3 / 180, formality lag 1 / H 3 / 126, constraint nu 4 / H 4 / 12376, CEFR lag 1 / H 4 / 4776; appendix columns are sentiment lag 1 / H 3 / 30, average word length lag 1 / H 3 / 42, defense nu 2 / H 3 / 180, GSM8K-sharded nu 4 / H 4 / 64. These are scored held-out rows after windowing, not trajectory counts, and the prose must say so. The settings differ per task, so a single sentence cannot cover them; they go in a small setup table or in the appendix, with 4.1 stating only the ranges

4.1 P3 (setup): control-line baselines are the four rows of Table 2 per frozen model, Base (no control, Access = --), TMPC (Access = white-box), RE-Control (Access = white-box), KOMPAS (Access = black-box) [tab_control; instantiates [choice_3]]; Access column states the requirement each method places on the model, latent-activation access for the two white-box controllers, nothing beyond an instruction for KOMPAS

4.1 P4 (setup, intuition + gloss): intuition, a predictor that never beats guessing the best fixed trivial answer scores zero and one that halves that guess's error scores one half; introduces the modeling-line metric [s_skill], prediction skill score, one minus the ratio of the operator's free-running rollout error to the best trivial null's rollout error over the same horizon [s_H]; glosses rollout error as the accumulated squared deviation between predicted and actual attribute across the horizon, best trivial null as the best-scoring memoryless baseline over the same rows; zero means tied with that null, negative means worse than it

4.1 P5 (setup): control-line metric is the constraint-satisfaction score as printed by the colleague, a per-subtask percentage with no interval attached [c5 notes; `numbers_benchmark.yaml` unit field]; TODO(data) STILL OPEN, NEEDS THE COLLEAGUE: the exact definition of the constraint-satisfaction metric, i.e. what counts as a satisfied instruction per subtask. `numbers_benchmark.yaml`'s own header records it as pending. Until it arrives the prose says "the constraint-satisfaction score as reported for each benchmark" and does not paraphrase what it measures; this is the last unresolved setup fact in the chapter; states plainly that the two lines report different metrics in different reference frames, the modeling line's skill score referenced against a best-trivial-null baseline and able to go negative, the control line's percentage unreferenced to any null point; the modeling line's job is testing whether the operator predicts the trajectory, the control line's job is testing whether acting on that prediction changes the outcome

### 4.2 Predictive Accuracy of the Learned Operator

4.2 P1 (RQ + Setup): **Question.** two-part yes/no, (a) does the operator need several turns of history to predict a multi-turn attribute well, and (b) does a nonlinear lifting or a sequence model predict better than the linear special case reported as Ours; setup sentence points back to 4.1 for tasks, baselines and the skill-score metric, no repetition (Rule 10); table pointer to Table 1 [tab_prediction]; (a) tests [choice_1] and [as_1], (b) tests [choice_2]'s linear instantiation and [as_2]

4.2 P2 (Findings, ranking and history): headline, the linear special case ranks first on four of seven columns (sentence length, items-s1, items-s2, CEFR), second on one (constraint), third on two (character length, formality) [c4; `_build_report.json:ours_best_of_7=4, ours_second_of_7=1`]; history beyond the current turn wins on five of seven columns with paired bootstrap intervals excluding zero, naming the two exceptions, items-s1 (interval containing zero) and formality (negative point estimate, interval containing zero) [c1; `_build_report.json:contrasts.ours_minus_last_turn`]; driver names the window as exposing level and trend together, which a single reading cannot carry [m1]

4.2 P3 (Findings, lifting and LSTM): a learned nonlinear lifting never beats the identity lifting with an interval excluding zero on any of the seven columns, and loses on four of them, largest gap on CEFR where the linear special case leads by +0.599 [c2; `_build_report.json:contrasts.ours_minus_nonlinear:tsar_cefr`]; driver names the observable space as already low-dimensional, leaving a lifting little structure left to expose [m4]; against an LSTM sequence baseline neither side dominates, two columns each with intervals excluding zero, items-s2 and CEFR favoring the linear special case, character length and constraint favoring the LSTM [c3; `_build_report.json:contrasts.ours_minus_lstm`]; reported as an observation, driver_keyword is "no mechanism established" for the two LSTM wins and none is invented here [c3]

4.2 P4 (closing): \uline{a multi-turn window beats the current reading on five of seven tracked attributes, answering (a) yes; neither a learned lifting nor an LSTM sequence model beats the reported linear special case on balance, the lifting losing outright and the LSTM splitting the seven columns evenly with no identified mechanism, answering (b) no} [c1, c2, c3, c4]

### 4.3 Constraint Satisfaction under Predictive Control

4.3 P1 (RQ + Setup): **Question.** does predictive control built on the learned operator raise constraint satisfaction over the same frozen model with no control and over the two white-box latent controllers; setup sentence points back to 4.1 for benchmarks, baselines and the constraint-satisfaction metric, no repetition (Rule 10); table pointer to Table 2 [tab_control]; tests [choice_3]

4.3 P2 (Findings, gain over Base): predictive control raises the eleven-subtask average constraint-satisfaction score over the same frozen model with no control by +14.5 points on Qwen3-4B and +14.8 on Qwen3-8B [c5; `_build_report.json:control_deltas."Qwen3-4B/Koopman - Base"=14.47, "Qwen3-8B/Koopman - Base"=14.76`]; driver names scoring on the predicted trajectory as what lets the controller pick a command that differs from the target while the attribute is still moving [m3]; the point-estimate status travels with the number in this same sentence as a property of the measurement, not a disclaimer, printed by the colleague with no seeds, no intervals, no independently specified metric definition yet [c5 notes; nc_3; D3; D12]

4.3 P3 (Findings, gain over white-box and sweep count): the same controller exceeds the stronger of the two white-box latent controllers, RE-Control, by +5.2 points on Qwen3-4B and +5.7 on Qwen3-8B, smallest per-subtask margin +2.4 and +2.7, both on the IFBench under-word-count subtask [c6; `_build_report.json:control_deltas."Qwen3-4B/Koopman - RE-Control"=5.25, "Qwen3-8B/Koopman - RE-Control"=5.65`]; driver names the access asymmetry as running the other way, RE-Control needing latent-activation access that KOMPAS does not [choice_3 notes]; across the two models and eleven subtasks the controller is best in all twenty-two cells, reported as a count rather than a significance statement given the same no-interval status as c5 and c6 [c7]

4.3 P4 (closing): \uline{predictive control raises constraint satisfaction over both an uncontrolled model and two white-box latent controllers, in every one of the twenty-two model-by-subtask cells measured, at the weaker access requirement of the two comparisons} [c5, c6, c7]

### 4.4 From Prediction Quality to Control Quality

4.4 P1 (RQ + Setup): **Question.** does prediction quality translate into control quality, and at what inference cost; setup sentence points to Method's Proposition 1 (section 3.4, proof in appendix A6) for the derivation, no repetition of it here (Rule 10 and Rule 21 boundary, Method owns the derivation, Experiments owns its consequence); no new dataset or metric introduced in this subsection

4.4 P2 (Findings, the bound): Proposition 1 of Method [eq_selection_bound] explains why more accurate rollouts keep the selected command's cost within twice the best achievable estimation error, so better prediction gives near-optimal candidate selection [c9]; theoretical, not measured, strength_basis is theoretical and no new number is planned here [c9]

4.4 P3 (Findings, the cost argument): evaluating a candidate costs one latent propagation and a decode via the closed-form rollout, and adds no query to the target model [c8; eq_closed_form_rollout]; driver names the affine transition plus the scalar command column as collapsing an h-step lookahead into a matrix power [m2]; structural property of the model, needs no measurement, and none is planned here [c8 notes]

4.4 P4 (Cross-check, equal-cost boundary): on the four self-built behavioral tasks the closed loop ties the best equal-cost fixed schedule [nc_2; appendix A4]; framed as where planning has headroom rather than as a failure, the boundary a property of what those tasks' command-response geometry looks like rather than of the controller; exact per-task numbers stay in appendix A4 (Rule 10)

4.4 P5 (Cross-check, command-channel scope): the command-channel point belongs only here, not in 4.2; on five of the seven modeling columns the applied command equals the tracking error, an exact affine function of the memory state, so the prediction design cannot estimate an independent action effect on those columns; on the remaining two columns (constraint, CEFR) the paired difference is +0.0015 and +0.0061 with intervals containing zero [m5; nc_1; nc_8; `_build_report.json:contrasts.ours_minus_memory_no_action`]; a property of how those trajectories were collected, not a result [m5 notes]

4.4 P6 (closing): \uline{better prediction keeps the selected command's cost within a bound that needs no extra LLM call to evaluate; the boundary is that on the self-built behavioral tasks planning only ties a fixed schedule, and the modeling experiments cannot see the command channel at all on most of their columns} [c8, c9, nc_1, nc_2]

### Missing setup facts

STATUS 2026-09-19 (Opus triage): three of the four were recoverable and have been filled in above
with their sources. One remains genuinely open and needs the colleague. The list below is kept as
the record of what was asked and how each was settled.

- TODO(data): which model produced the self-built multi-turn trajectories used by the modeling line. Not named in contract.yaml, semantic.md's Introduction or Method blocks, `paper/tables/_build_report.json`, `paper/tables/tab_prediction.tex`, or `paper/evidence/numbers_benchmark.yaml`.
- TODO(data): how many trajectories each self-built task has. Not stated in any file this outline's read list covers; `paper/evidence/numbers.yaml` (outside the specified read set) carries per-row "scored rows" counts, but those are post-windowing row counts, not a stated trajectory count, and were not used here.
- TODO(data): the horizon (and lag) used per self-built task. Not stated in contract.yaml, the two tables, or the story framework; per-row annotations in `paper/evidence/numbers.yaml` (outside the specified read set) show values that were not confirmed uniform across all eleven tasks, so none is asserted here.
- TODO(data): the exact definition of the control line's constraint-satisfaction metric, i.e. what counts as a satisfied instruction per subtask. `paper/evidence/numbers_benchmark.yaml`'s own header states this definition is "pending from the colleague."
