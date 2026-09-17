# Tier 2 — Semantic outline for *When Not to Close the Loop: A One-Sided Criterion for Multi-Turn LLM Behavior Control*

Built section-by-section in the A9 order; bullets are operational instructions referencing
`paper/contract.yaml` IDs, never draft prose (A2); page budget NeurIPS 9 pages.

Allocation so far: §2 = 6 paragraphs against a 0.5–0.75 page target, five of them short.
Revised after the Tier-2 cold review: the third named assumption was homeless, the definition
bullet was carrying enough to split at Tier 3, and the bridge was a trailing clause that
compression would have eaten first.

---

## §2 Problem Formulation

  P1 (setup): frames the regime — a multi-turn exchange, one scalar behavioral readout per turn
    scored by an independent judge, one intervention issuable per turn under a fixed budget;
    introduces [s4] informally and WITHOUT an equation; names the equal-cost fixed schedule as the
    comparison the whole paper is measured against, since [c5, c6] and [choice_4] all rest on it;
    scopes to the four task families and blocks the general reading via [nc_3]; constraints: no
    math, no symbols beyond naming the intervention, ≤4 sentences

  P2 (intuition): instantiates [ex_1] so that one example covers both sides of the split — the
    present level selecting the next instruction, and the instruction's effect being indifferent
    to when it is issued; motivates [prior_1, prior_2]; constraints: Rule 22 grade — no math, no
    numbers, no symbols; anchor example has its single prose home here and recurs nowhere else;
    ≤4 sentences

  P3 (assumption): states all three named assumptions as one numbered block — [as_1] at strength
    `tendency`, [as_2] at strength `approximate`, [as_3] at strength `tendency` — each as a plain
    statement BEFORE its formal pointer (A11); flags that [as_2] is the one the paper measures as
    false on every column and that reporting this is the second finding rather than a defect in the
    framing; flags that [as_3] is checked on one side only, pointing at [nc_2]; constraints: no
    equations, one assumption per sentence, each stated at its contract strength and nowhere
    restated at a different one

  P4 (definition): introduces [eq_1] and defines [s1, s2, s3, s5, s6]; realizes [as_1]'s formal
    pointer; constraints: Rule 14a — every symbol glossed at first use in the same sentence; the
    core line's rank deficiency is NOT stated here, it belongs to §3.2 per [choice_1] notes

  P5 (definition): introduces [eq_9] as the reported quantity and defines [s16, s17]; states the
    two cases where the metric refuses to return a number and why each is load-bearing; states
    that the quantity is not comparable across columns, so the paper carries no aggregate row
    anywhere; constraints: the undefined-versus-zero distinction survives any compression, because
    it is the entire content of the negative-control column

  P6 (bridge): points to §3 and names which assumption each Method subsection will formalize;
    constraints: ≤1 sentence, no new symbols

---

## §3 Method

Allocation: 1 overview + 4 module subsections = 12 paragraphs. Revised twice. The Tier-2 cold
review found the depth trade inverted, so the procedural subsections got a How-X each and §3.2
dropped to two paragraphs because [eq_1] is already displayed in §2 and is recalled here rather
than reintroduced. Then the page projection came in 0.9–2.6 pages over a 9-page limit, and the
user moved [choice_3] to the appendix: it estimates the BASELINE rather than the operator this
paper is about, so it had the weakest claim on main-text space among the five. It keeps its
rq4 test and is named once in the overview; [eq_2, eq_3, eq_4, eq_7] and [s7, s8, s11, s18] move
with it, leaving §3 with three displayed equations. Rule 24 holds
throughout: exactly one Why-X per module subsection, none in the overview. Rule 21 holds
throughout: no measured number enters this section. That bites twice here, because three of the
notes in `equations.tex` carry measured values and because every `choice_N` carries a numeric
`finding_keyword` that a drafter will read as the stage's payoff; both belong to §4.
[m5] and [m7] carry `appears_in_method: false` and are allocated in §4, so their absence here is
by design rather than a dead contract.

### §3.1 Framework Overview

  P1 (overview): walking tour of the four stages in pipeline order — build the state [choice_1],
    fit the operator with its action channel [choice_2], estimate it [choice_3], then either
    search actions against it [choice_4] or read it as a pre-investment test [choice_5]; names
    [choice_3] once and routes it to the appendix, since it estimates the baseline rather than the
    operator; anchors
    on the framework figure; poses the section-level question that §3.5's closing answers, which
    is the only question this paragraph carries; constraints: no equations, no symbols, no Why-X
    (Rule 24), one paragraph, each stage named by its subsection title verbatim; Rule 21 — the
    `finding_keyword` of every choice is §4 material and none of it appears here

### §3.2 Delay-Embedded Controlled State

  P1 (Why-X): motivates both facets of [choice_1]'s title — why a window rather than a turn, and
    why the action enters the same vector rather than a side input; realizes [as_1] and [m1];
    contrasts the memoryless alternative; constraints: Rule 22 — no formulas, no symbols, no
    numbers; ends with one underlined answer naming both facets (Rule 18)

  P2 (How-X): recalls [eq_1] and [s6] from §2 without reintroducing either, since [eq_1] is
    `in_section: problem_formulation` and §2 P4 is its display home; walks the row-eligibility
    rule; states the structural consequence on the trajectory-tracking line, that its control is a
    function of its own state and so that line cannot separate an action effect from a state
    effect; names [m2] as the competing account for the memory claim that this construction alone
    cannot rule out, deferring its test to §4; hands off to §3.3; constraints: Rule 21 — the
    rank-deficiency enters as a structural property and its measured value stays in §4; no depth
    values from [choice_1]'s `finding_keyword`

### §3.3 Linear Operator with an Explicit Action Channel

  P1 (Why-X): motivates both facets of [choice_2]'s title — why linear, and why the action is its
    own regressor rather than folded into the state; realizes [as_2] and [m3]; states that this
    construction is what turns withholding the action into a measurable ablation rather than a
    modeling opinion; constraints: Rule 22; ends with one underlined answer naming both facets

  P2 (Eq + gloss): displays [eq_5] realizing [as_2]'s formal pointer; defines [s9, s10] and
    recalls [s4] from §2; introduces the interaction-lifted control variant; states that the
    readout is a second, separate ridge problem rather than a joint fit; constraints: Rule 14a
    coverage over every RHS symbol including the recalled ones

  P3 (How-X): defines the withheld-action variant as an object, the same estimator with the
    control column removed, and names its contrast against the full fit as the quantity §4 reports
    under [eq_9] from §2; bridges to §3.4 by noting that the same contrast is defined for the
    learned-lifting estimator; constraints: Rule 21 — no contrast value here

### §3.4 Budget-Pruned Action Search

  P1 (Why-X): motivates both facets of [choice_4]'s title — why search over action sequences at
    all, and why the budget prunes the tree rather than entering as a penalty; realizes [m3];
    names the equal-cost fixed schedule from §2 as the comparison this controller is built to be
    measured against; constraints: Rule 22; ends with one underlined answer naming both facets

  P2 (Eq + gloss): displays [eq_6]; defines [s12, s13] and recalls [s4]; constraints: Rule 14a
    coverage over every RHS symbol including the recalled ones

  P3 (How-X): walks the recursion and states the red line, that the admissible-set collapse is the
    entire constraint mechanism, so no Lagrangian, projection or relaxation language may describe
    this controller; gives the budget-one specialization as the two-way comparison the recursion
    reduces to; references [eq_7] as a closed-form special case that coincides with the search only
    when both branches leave the second step idle, marked a remark rather than the decision rule
    and displayed in the appendix; bridges to §3.5

### §3.5 Operator Diagnostics as a Pre-Investment Test

  P1 (Why-X): motivates both facets of [choice_5]'s title — why the identification fit is the
    right place to ask whether control can pay, and why the answer is available before the
    closed-loop compute is spent; realizes [as_3] and [m4]; states the one-sidedness here rather
    than deferring it, recording [nc_2]; constraints: Rule 22; ends with one underlined answer
    naming both facets and the one-sidedness

  P2 (Eq + gloss): displays [eq_8] realizing [as_3]'s formal pointer; defines [s14, s15]; states
    that these are fit diagnostics carrying no cross-arm comparison; constraints: Rule 14a
    coverage; Rule 21 — the half-life-versus-payoff mismatch and the impulse-length instrument
    defect are §4 and §5 material and are not stated here

  P3 (closing): integrates the five subsections into the one test the paper sells and hands off to
    §4 by naming which RQ tests which subsection; constraints: ≤3 sentences, no new symbols; its
    underlined answer closes the section-level question posed in §3.1 P1, not §3.5 P1's, which its
    own Why-X already answered

### Appendix A1 (Reconstruction-Then-Ridge Estimation)

  P1 (Why-X): motivates both facets of [choice_3]'s title — why representation and dynamics are
    fitted in separate stages, and why the intercept is left unpenalized; realizes [m6]; frames
    the learned lifting as the alternative this paper fits fairly rather than as a strawman;
    constraints: Rule 22; ends with one underlined answer naming both facets

  P2 (Eq + gloss): displays [eq_2] for the lift the baseline estimates and [eq_3] for the
    two-stage solve; defines [s7, s8, s11, s18]; constraints: Rule 14a coverage; two displayed
    equations is this subsection's ceiling

  P3 (How-X): walks the two stages in order and states the zeroed intercept penalty as its own
    sentence, since no comment in the released code marks it and a penalized intercept would make
    this a different estimator; references [eq_4] in prose with display deferred to the appendix,
    and records [nc_6] as the scope boundary that its multi-step term runs on a separate optimizer
    schedule and was never swept at comparable strength; bridges to Appendix A2; constraints: [nc_6] is a
    scope boundary, not a result

  note: [choice_3] keeps its Why-X, its displayed equations [eq_2, eq_3] and its How-X here;
    [eq_4] and [eq_7] are displayed in this appendix too. Moved out of §3 under the page budget
    by user decision 2026-09-17, not because the choice stopped being defended.
