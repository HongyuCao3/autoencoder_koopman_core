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
