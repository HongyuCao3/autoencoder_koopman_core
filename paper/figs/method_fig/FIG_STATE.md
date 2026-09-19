# FIG_STATE (method figure, TikZ)

## v3 (post CP2-redo)

updated: 2026-09-19  by: sonnet session (v3-0 through v3-2, this session)
step: v3-0 prepared -> v3-1 (B/C/D redrawn) -> v3-2 (right_panel assembled) -- see
  step log below for exact status of each.
context: `paper/sections/02_method.tex` was rewritten (CP2-redo, commit `6be3d2b`)
  between step-3-rev-4 (the last state recorded above) and this session. The
  rewrite drops the xi-space encoder/decoder and the three-term training loss
  from the main text (they now live only in the appendix) and moves the model
  onto the raw reading window s_t with a one-line least-squares fit. See
  `FIG_PLAN_v3_2026-09-19.md` for the full plan this session executed; do not
  re-derive requirements from `FIG_PLAN_2026-09-19.md` (v2) except where v3
  explicitly says something is unchanged (module A, left_loop, the 2x2 layout,
  the glyph vocabulary, kompas_style.tex's palette/styles).

### v3-0 (shared files)
- `symbols.tex`: removed the macros the plan lists (`\symLat`, `\symLatNext`,
  `\symLatHat`, `\symLatDim`, `\symDec`, `\symQ`, `\symReadout`, `\symLatRoll`,
  `\symData`, `\symLrec`, `\symLlin`, `\symLpred`, `\symParams`, the xi-space
  `\eqAffine`). Kept `\symEnc` (E_phi appears once in 3.1 P6 and once in
  module B). Added exactly the macros the plan specifies: `\symMemRoll`,
  `\symKhat`/`\symBhat`/`\symbhat`, `\symFit`, `\symTrans`, `\symResid`,
  `\symBudget`, the s-space `\eqAffine`, and `\eqIdent`.
- `kompas_style.tex`: appended exactly the two macros the plan specifies,
  `\MemColVF` and `\WinBracket`, at the end of the file, under a new
  "v3 additions" banner comment. Nothing else in this file was touched --
  palette, `panel`/`ptag`/`thinflow`/`oursflow`/`bypass`/`lossline` styles and
  the `clModA`-`clModD` colours are all byte-for-byte what step-3-rev-4 left.
- Predicted-failure check (plan says B/C/D SHOULD fail G2 at this point,
  before their redraw, because they still reference macros just deleted):
  ```
  modA_memory.tex: OK (4 atoms, 0 unmatched)
  modB_koopman.tex: 4 UNMATCHED -> \symDec, \symLat, \symLatDim, \symLatHat
  modC_training.tex: 10 UNMATCHED -> \symData, \symDec, \symLat, \symLatNext,
    \symLlin, \symLpred, \symLrec, \symParams, \symQ, \theta
  modD_control.tex: 3 UNMATCHED -> \symLat, \symLatRoll, \symReadout
  ```
  (These read as literal macro names, not expanded LaTeX, because the audit
  script's macro table no longer has entries for them -- exactly the signal
  the plan says to expect. B/C/D would also now fail to *compile*, for the
  same reason: undefined control sequences. This is the expected, momentary
  broken state between deleting the old symbols and redrawing the three
  modules that used them.)

### v3-1 (B, C, D redrawn) -- DONE, this session

Backups made before any editing (per this task's own instructions, not part
of the v3 plan's own step list): `paper/figs/method_fig_v2_backup_2026-09-19/`
is a full copy of the pre-v3 `method_fig/` directory (the step-3-rev-4 state,
never approved, never wired in), and
`paper/figs/_colleague_backup_2026-09-18/method.png` is a copy of the
colleague's raster figure that `02_method.tex` still `\includegraphics`'s
(no such backup existed before this session).

**modB_koopman.tex** -- full rewrite. Main chain, baseline y=1.75, entirely in
s-space: `s_t -> [K, (L+1)x(L+1)] -> (+) -> shat_{t+1} -> [C] -> yhat_{t+1}`,
command (B, ours) entering from below, bias (b, small, labelled to its west)
from above. The lifting is one faded (opacity 0.45) arc from s_t's own column
top back to shat_{t+1}'s own column top, through a de-emphasised encoder/
decoder pair positioned in the gaps flanking the busy middle (over the s_t->K
arrow and over shat_{t+1}), not directly over K or the bias column -- three
earlier placements of that arc each produced a different text collision only
visible in the render (E_phi merging into K's own (L+1)x(L+1) brace label;
"command"/$B$ merging into "comBand" once the command column grew from the
old 3 rows to the correct 4; the net colliding with the bias column's own
top). `\symEnc` (E_phi) appears exactly once, under the encoder, as required.
Deleted: both `\IdBypass` arrows, both `\Trainable` frames, "(feeds D)".
One exemption logged (plan's own §8 risk table anticipated this class of
bug for `\symResid`, and it recurred here for a different pair): the G2
audit's naive string-substitution macro expander reads `\times` immediately
followed by `\symMemDim`'s own expansion "L+1" as the literal nonexistent
control word `\timesL`, even though real TeX tokenizes them as two separate
control words. Fixed with a no-op `{}` between them
(`\symMemDim\times{}\symMemDim`) -- changes nothing rendered, silences the
false positive; the script itself was not touched, per the plan's own
prescription for this class of issue.
Gates: G1 pass (`preview/step_modB_koopman_v21.png`, final). G2 pass (10
atoms, 0 unmatched). G3 pass (visual, iterated v15->v21, see history in
this session's tool log if ever needed -- not reproduced here). G4 pass (2
minicap nodes: "command", "optional learned lifting"; 0 banned verbs). G5
pass (module colour only in the title bar, drawn in right_panel.tex). G7:
one `line width=0.5pt` hit, on the sum-node circle -- carried over verbatim
from the step-3-rev-4 version of this module (already gated pass then; not
a new override). G8 pass (0 hits). G10 pass (`\symEnc` count = 1, in B).
G11 pass (all main-chain horizontal arrows at y=1.75).

**modC_training.tex -> modC_learning.tex** -- renamed (old file deleted from
the working tree; it is still recoverable from
`method_fig_v2_backup_2026-09-19/` and from git history) and rewritten.
Left half draws 3.2 P1 (the training set is overlapping windows sliced from
a recorded reading stream): one `\MemRow`, two `\WinBracket`s offset by one
cell, an arrow down to three staggered small transition cards
`(s_t, c_t, s_{t+1})` (only the front card labelled, via `\symTrans`). Right
half draws eq:identification: the predicted side `K s_t + B c_t + b` in the
SAME blocks and colours module B uses (cell shrunk to 0.16, per spec), one
`\LossLink` (not two crossing ones) against the recorded `s_{t+1}` column,
labelled `\symResid` at its midpoint; the fit `(Khat, Bhat, bhat)` drops out
below via one `thinflow` arrow, as three small blocks with the hatted symbol
below each. "theta frozen" is `\SnowIcon` + `\symLLM`, not a text caption.
Zero `\EncNet`/`\DecNet`, zero `L_rec`/`L_lin`/`L_pred`, zero `q_t`, one
lossline (was two crossing ones) -- this was the module the plan flagged as
the largest win, and it reads far cleaner than the retired version now.
Two collisions only visible in the render, both fixed by relocating a label
rather than by recomputing coordinates on paper: `\symTrans`'s own bounding
box reaches lower than a plain scriptsize line would (its triple subscript
`(s_t,c_t,s_{t+1})`), so the first "sliding window" minicap position merged
with it; and `\symMemNext`'s label, first placed above the recorded column,
sat close enough to "least squares, closed form" to read as one run-on
string, fixed by moving `\symMemNext` to above the column's own top instead
(same trick module B already uses for `\symMemHat`) and nudging the minicap
left.
Gates: G1 pass (`preview/step_modC_learning_v4.png`, final). G2 pass (8
atoms, 0 unmatched). G3 pass (visual, iterated v1->v4). G4 pass (2 minicap
nodes: "sliding window", "least squares, closed form"; 0 banned verbs). G5
pass. G7 pass (0 hits). G8 pass (0 hits in prose/comments; `clLatentD` -- a
pre-existing kompas_style.tex colour-macro NAME shared with module B's K
block, not new to this module -- matches the G8 grep's `latent` pattern
case-insensitively but is not the banned term in the sense G8 means; logged
as an accepted false positive rather than renaming a shared palette colour,
which the plan's "no new colours" rule would treat as scope creep anyway).
G10 pass (0 occurrences of `\symEnc` -- this module never lifts). G11 pass.

**modD_control.tex** -- middle section rewritten, both ends kept exactly as
specified. Section 1 (candidate set `C_t`) and section 4/5 (predicted-
trajectory-vs-target plot, cost bars, argmin) are untouched. Section 2
(rollout) now runs on `s_t` via `\MemColV`/`\MemColVF` (was `\LatCol`/
`\LatColVF` on `xi_t`), baseline moved to y=1.75 (was 1.93), column spacing
widened (now >=0.75cm centre-to-centre) since the decoder that used to eat
the middle of this module is gone. Section 3 (readout) is now one
`\RowSelector` (labelled `\symSel`), not a `\DecNet` + `\symReadout`. Section
6 (the receding-horizon loop) keeps its three-block shape, only the labels
changed: Execute/Observe/Replan -> Send/Read/Replan, so this loop, the left
column and Algorithm 1 all use the same three verbs. One collision only
visible in the render: `\symMemRoll` first placed below the rollout chain
merged with the `h=1,...,H` brace label directly underneath it (that label's
own subscript stack makes it wider than a coordinate estimate suggests);
fixed by moving it above the chain's own top instead, mirroring the
`\symMemHat` / `\symMemNext` fix used in B and C for the same class of bug.
Gates: G1 pass (`preview/step_modD_control_v22.png`, final). G2 pass (12
atoms, 0 unmatched). G3 pass (visual, iterated v20->v22). G4 pass (2 minicap
nodes: "no LLM call", `\symBound`; 0 banned verbs). G5 pass. G7: the
hand-drawn `line width=...` hits are all in the untouched candidate-set and
plot sections, carried over verbatim from step-3-rev-4 (already gated pass
then). G8: one hit on first pass, in this session's OWN new top-of-file
comment ("not a latent xi_t") -- reworded to describe the same fact without
the banned word ("the appendix's learned-feature variant is not drawn
here"); 0 hits after. G10 pass (0 occurrences of `\symEnc`). G11 pass (all
new rollout-chain arrows at y=1.75).

### v3-2 (right_panel assembled) -- DONE, this session

**modA_memory.tex** -- no changes. Verified against §3.1: content matches
("内容不变"), and the `s_t` `\MemColV` is still top-left-anchored at local
y=2.09 (unchanged), which is what the cross-module arrow depends on. The
plan's optional mini-plot tweak (swap the two trend lines' end-dots for
`\ScalarCell`s) was left as-is (filled dots) -- it is explicitly a
take-it-or-leave-it choice in the plan, not a requirement, and there was no
reason on inspection to prefer the alternative.

**right_panel.tex**:
- Title bar text updated to the plan's exact §2.2 wording: "A Finite-memory
  state" / "B Koopman dynamics model" / "C Learning from data" / "D
  Predictive selection".
- Deviation, logged here: panel A's title, at one line and the shared
  \scriptsize title font, clipped against its own panel's rounded corner --
  A's width (2.94cm) is unchanged and is the narrowest of the four, and its
  new required text is longer than the old "A Finite memory". G3 sets
  \scriptsize as a FLOOR ("minimum font size"), so shrinking the font (tried
  \tiny first) was rejected as it would violate that gate. Fixed instead by
  wrapping A's title onto two lines ("Finite-memory\\state") and adding
  `align=center` to the shared `\PanelTitle` macro (applies uniformly to all
  four titles, so this is one shared-macro change, not a per-panel style
  override) plus growing the shared `\TitleH` from 0.36 to 0.46cm (also
  uniform across all four panels) to fit the second line. B/C/D's titles
  still render on one line at the same \scriptsize; only A wraps.
- Deleted the "(feeds D)" node and its surrounding comment block. Per the
  plan, this is not replaced with anything (no line, no new caption): B's
  `\symYhat` and D's rollout/readout are the same glyphs, same colours, same
  symbols, and that shared identity is what the plan wants carrying the
  relation, once every other main-text-only object is gone from the figure.
- `\input{modC_training}` -> `\input{modC_learning}`.
- Reworded the surviving cross-module-arrow comment, which used the word
  "aligned" (matches G8's `align` pattern) -- not a new violation (that
  comment predates v3), but since this file was already being edited this
  session, it is fixed here rather than left for a future G8 run to catch.
- Verified: the A->B cross-module arrow (`(1.91,{2.09+\RowOneY})` --
  `(3.81,{2.09+\RowOneY})`) is unchanged and still the only cross-module
  `\draw` in the file (G9).

Gates on the full assembly: G1 pass (`preview/step_right_panel_v22.png`,
CURRENT, supersedes v20/v21 and all step-3-rev-4 renders). G2 pass, run
across all five fragments together (`modA_memory.tex modB_koopman.tex
modC_learning.tex modD_control.tex right_panel.tex`): every file 0
unmatched. G8 pass on `right_panel.tex` itself (0 hits after the reword
above). G9 pass (exactly 1 cross-module arrow, A->B). G10 pass (`grep -c
symEnc mod*.tex` = 1, and it is in `modB_koopman.tex`).

previews_for_review (v3, current):
  preview/step_modB_koopman_v21.png
  preview/step_modC_learning_v4.png
  preview/step_modD_control_v22.png
  preview/step_right_panel_v22.png   <- the one the user should look at first
user_verdict_on_previous: not yet recorded -- v3-2 is exactly the plan's own
  stop point ("停：用户看右面板 PNG"). v3-3 (assemble method_figure.tex, wire
  it into 02_method.tex's `\includegraphics`, rewrite the caption and the
  opening-paragraph reference, run mech_audit.sh / latexmk) is NOT done and
  was explicitly out of scope for this session -- do not do it until the
  user has approved this right-panel render.

updated: 2026-09-19  by: sonnet session (step-3 rev 4: caption trims, D's loop as 3 blocks, C redrawn left-to-right, this session)
step: 3 (rev 4)   status: awaiting_user_review
decisions: Q1=C-as-own-module (user, 2026-09-19, 2x2 保持四块)
           Q2=sentence-length (user, 默认)
           Q3=no-bound-glyph, minicap only (user, 默认)
gates (step 1, left_loop rev2, carried over from prior entry):
       compile=pass
       symbol_audit=n/a (left_loop.tex 0 atoms after rev2 removed all formulas;
                          notation consistency now rests entirely on the 4 right modules)
       visual_check=pass(opus viewed step_left_loop_v6.png)
       G3/G4/G5/G6 = pass (per prior entry)
gates (step 2, four modules, this session):
       compile=pass (all 4: modA v5, modB v10, modC v10, modD v8)
       symbol_audit (G2, python paper/tools/fig_symbol_audit.py):
         modA_memory.tex  OK  (4 atoms, 0 unmatched)
         modB_koopman.tex OK  (15 atoms, 0 unmatched)
         modD_control.tex OK  (12 atoms, 0 unmatched)
         modC_training.tex FAIL (7 unmatched: \mathcal{D}, \mathcal{L}_{rec/lin/pred},
           \phi, \psi, \theta as bare atoms) -- root cause: this is the known
           D19-vs-Q1 conflict already logged below (training objective was moved
           to Appendix A1_estimator.tex; 02_method.tex itself no longer states
           L_rec/L_lin/L_pred or a bare \mathcal{D}). The script only checks
           02_method.tex, so module C's G2 gate cannot mechanically pass while
           C stays in the main figure under the user's Q1 ruling. Not a new
           notation drift (verified by grep: no other spelling appears for these
           symbols anywhere in sections/); flagged, not fixed, pending Phase 2
           decision on whether the audit script should also check appendix/A1.
       visual_check=pass (this session viewed step_modA_memory_v5.png,
         step_modB_koopman_v10.png, step_modC_training_v10.png,
         step_modD_control_v8.png)
       G3 no overlap / arrows on borders / min \scriptsize = pass (visual)
       G4 <=4 words per node, no sentences, no result numbers in labels = pass (visual)
       G5 colour semantics (blue=structure, ours-blue=command path/argmin,
          green=target/good, red=loss link/overshoot) = pass (visual)
       G6 no experiment-result numbers, only example symbols (L+1, d_xi, H, h) = pass
       G7 text-only-node grep (node[...]{...} with no glyph) = 0 for all 4 modules;
          no stray \tikzset / global font or line-width override found (per-line
          \draw[line width=...] for hand-drawn plot lines and connectors is
          expected per FIG_PLAN 2a/2d spec, not a style override)
previews_for_review: preview/step_left_loop_v6.png (step 1, rev 2),
                     preview/step_modA_memory_v3.png,
                     preview/step_modB_koopman_v13.png (rebuilt after the
                       y-hat label move below; content otherwise identical
                       to v12),
                     preview/step_modC_training_v14.png,
                     preview/step_modD_control_v17.png,
                     preview/step_right_panel_v20.png (step 3 rev 4, CURRENT
                       -- supersedes v12: full-width centred title bars on
                       all 4 panels (was: small corner tags), plus the one
                       text-overlap bug this surfaced, see the note below)
user_verdict_on_previous: not yet recorded in this file for step 1 rev2 or step 2 --
  rev2's changes already encode detailed user feedback from a prior review (see
  notes below), but no explicit approved/revise verdict was logged before the
  same session went on to draft all four step-2 modules without commit/state
  update. This entry is a bookkeeping catch-up: it records the true on-disk
  state (step 2 fully drafted, gates run, nothing committed) so a fresh session
  is not misled by the stale step:1 entry that preceded it. All 5 images above
  were just sent to the user for review together.
next_action: step 3 rev 4 (caption trims + D's loop as blocks + C redrawn
             left-to-right) is drafted, rendered, gate-checked and committed;
             still awaiting a user verdict on step 1 rev2 + step 2 (four
             modules) + step 3 (panel, now rev 4) together -- no verdict has
             been given on ANY version of step 3 yet. Do NOT start step 4
             (master assembly, wire into 02_method.tex) before that verdict
             arrives. On approve -> step 4. On revise -> iterate the named
             module/panel only, bump version, re-run gates.
notes:
  - THIS SESSION, step 3 rev 2 (user feedback: "有一些文字overlap问题需要解决,
    神经网络设计元素中的圆形半径减小,因为现在重叠比较严重"):
    ROOT CAUSE of the neuron overlap (affects every scaled EncNet/DecNet
    instance, i.e. all of B, C x4, D): TikZ's `scale` on a `\begin{scope}`
    shrinks coordinates but NOT a node's `minimum size` unless the scope also
    sets `transform shape`. The 7 call sites (modB x2 @0.62, modC x4
    @0.40/0.40/0.40/0.30, modD x1 @0.26) were missing it, so the neuron
    circles stayed at their full 2.5mm diameter while the trapezoid and the
    inter-neuron spacing shrank around them -- worse at smaller scale, which
    is exactly the pattern the user reported. Fixed by adding
    `,transform shape` to all 7 scopes, plus shrank the base neuron
    `minimum size` 2.5mm -> 2.0mm in kompas_style.tex for extra margin (the
    literal "reduce the radius" ask). Confirmed by re-rendering and cropping
    B/C/D's nets at 3-8x zoom: all neurons now read as distinct circles.
    TEXT-OVERLAP fixes, each found by cropping+zooming the actual render
    (not by re-deriving coordinates on paper -- that approach was tried
    first and got the diagnosis wrong twice before the crop caught the real
    node; left in git history as a record of what NOT to do next time):
      1. D: my own step-3 "model: from B" caption sat on top of D's own
         xi_t label. Deleted; replaced with "(feeds D)" inside B's own scope,
         below its y-hat cell (genuinely empty margin there), so the note
         reads at the source instead of colliding at the destination.
      2. C: "trains B" caption was effectively glued to the tag's own box
         (0.03cm clearance) and set in the tag's own gold colour (near
         -invisible against it). DROPPED rather than re-positioned: C's own
         bottom caption "updates (phi,psi,K,B,b); theta frozen" already
         names B as a trained parameter, so the relation is stated without
         adding a label that collides with either the tag or the s_t glyph
         (checked: C's top strip has no genuinely free band -- tag on the
         left, q_t/D_psi(q_t) labels on the right).
      3. D's readout cluster (decoder -> row-selector -> g_psi, the
         eq:rollout readout once per FIG_PLAN 2d): three compounding bugs,
         found one at a time by re-cropping after each fix --
         (a) the decoder was drawn at scale 0.26, half of FIG_PLAN \S3.2's
             own "50%" spec, which is why everything downstream was too
             tight to begin with. Grown to 0.35 and every arrow/selector
             coordinate recomputed from its actual bounding box.
         (b) thinflow's 1.7mm arrowhead is sized for the main chain; next to
             a net this small it was wider than the glyph itself. Added a
             new `microflow` style (0.9mm head) in kompas_style.tex and used
             it for the three arrows touching this cluster.
         (c) the actual worst offender, found only after (a) and (b) still
             looked wrong under an 800dpi crop: the mini-axes' own y-roll
             label (section 4 of the file, unrelated to the readout cluster
             in section 3) was anchor=south-east, which grows LEFT/UP from
             its point and was reaching back into the decoder from across
             the module. Every earlier fix in this cluster left it untouched
             because it lives in a different part of the file. Re-anchored
             south-west so it grows away from the decoder instead.
      4. C: the middle decoder's (mcD1) own $D_\psi$ tag was anchor=south at
         a y already inside the trapezoid's own vertical span (the one label
         in the file not following the sibling anchor=north-below-the-shape
         convention used by mcE1/mcE2/mcD3). Moving it to anchor=north below
         the shape only traded this collision for another -- that band sits
         exactly where the L_lin/L_pred loss circles are -- so DROPPED it
         instead: this decoder is already named by the labelled encoder
         right before it in the same chain and by the "D_psi(q_t)" caption
         directly above it, so the third, colliding copy loses no
         information.
    HOUSEKEEPING found along the way: two of my own inline .tex comments
    contained literal $...$ math (e.g. explaining a label via "$\hat\xi_{t+
    h|t}$" in a % comment); fig_symbol_audit.py does not strip TeX comments,
    so these were briefly flagged as new unmatched symbols. Rewrote the
    comments in plain text. Re-ran G2 on all 5 fragments after every change
    in this note: modA/B/D/right_panel all OK 0 unmatched; modC still shows
    its known 7 (the D19-vs-Q1 appendix gap logged above), unchanged by any
    edit in this pass.
    Final artefact for review: preview/step_right_panel_v12.png (also
    step_modA_memory_v3, step_modB_koopman_v12, step_modC_training_v14,
    step_modD_control_v17 -- same content as v11/v13/v16 respectively, just
    recompiled after the comment cleanup, no visual change).
  - THIS SESSION, step 3 rev 3 (user feedback: "ABCD四个块的标题改成和矩形等长
    文字居中，也调整其他图像位置避免标题遮盖其他内容" -- titles must span the
    full width of each panel, centred, and nothing may sit under them):
    Rather than reflow any of the four modules' own (already tightly-packed,
    already-gated) internal coordinate systems to make room for a full-width
    bar, added a new strip ABOVE each panel's existing content, exclusively
    for the title: `\TitleH=0.36` (bar height), `\PanelH=\ModH+\TitleH=3.39`
    (new full panel height), `\RowOneY=\PanelH+0.25=3.64` (new row-1 y-offset,
    replacing the old literal 3.28). New `\PanelTitle` macro draws a filled
    rounded-rect spanning the panel's own width with centred bold white text.
    Every module's own local (0,0) origin is unchanged -- only the panel box
    and the new title strip grew around it -- so no modA/C/D edits were
    needed for the redesign itself. Verified by cropping each of the 4
    title-strip-to-content transitions at 500dpi: all 4 titles read centred
    and full-width, all 4 have a visible content gap below the bar, and the
    inter-row gap (row1 bottom border to row2 title top) is unaffected.
    ONE real overlap surfaced by this change, found the same way as every
    other overlap this session (crop+zoom the actual render, not paper
    coordinates): B's "(feeds D)" caption only needed a uniform +\RowOneY
    shift (same as everything else in row 1), so on paper it looked
    unaffected -- but at 600dpi it was sitting directly on top of B's own
    $\hat y_{t+1}$ label (both objects centred at the same local x=5.75; the
    caption's top edge at y=0.45 landed inside the label's own vertical
    span). This was not a new bug from the title change; the same collision
    would already have existed in v12 as soon as anyone zoomed that corner
    -- it had simply never been crop-checked at high enough resolution
    before now, and the title-height change is what prompted a full re-crop
    of every corner of the panel this time. Fixed in two parts:
      1. modB_koopman.tex: the $\hat y_{t+1}$ label was coming from
         \ScalarCell's built-in south-of-the-box placement, which put it
         directly below the box with no margin to spare. Replaced the macro
         call with the same box drawn by hand plus a custom label anchored
         WEST of the box instead of south, freeing the strip below the box
         entirely (module-internal edit, needed because the label's own
         default position was the root cause, not anything in right_panel.tex).
      2. right_panel.tex: moved the "(feeds D)" node to sit below the box in
         the now-empty strip (local y 0.45 -> 0.28) and re-anchored it
         north-west starting at the box's own right edge (local x 5.75 ->
         5.86) instead of centring it under the box, which also fixed a
         second, smaller issue the first fix's geometry would otherwise have
         left behind: at the lower y, "(feeds D)" was landing edge-to-edge
         against B's own "command enters here" caption with no visible gap.
    Re-ran G2 on modB_koopman.tex and right_panel.tex after both edits: OK,
    0 unmatched (the label swap and caption move are geometry-only, no
    symbol changed). No other module touched.
    HOUSEKEEPING found along the way (same class of bug as the step-3-rev-2
    entry above): fig_symbol_audit.py treats any $...$ span as math mode, but
    TikZ's calc library also delimits coordinate arithmetic with $...$; the
    new \PanelTitle macro used calc syntax to add \TitleH to a coordinate
    (`($#1+(#2,\TitleH)$)`), so the audit read \TitleH as an undefined paper
    symbol and failed G2 on right_panel.tex (1 unmatched). \TitleH is a
    layout constant, not notation, so this was a false positive, not a real
    gap. Fixed by rewriting \PanelBox/\PanelTitle to do their arithmetic in a
    shifted scope with plain brace-pgfmath coordinates (the `{2.09+\RowOneY}`
    style already used elsewhere in this file) instead of calc's $...$ --
    zero visual change (confirmed: v16 is byte-identical to v15), G2 now
    OK (11 atoms, 0 unmatched).
    Final artefact for review: preview/step_right_panel_v16.png (also
    step_modB_koopman_v13.png, same content as the version reviewed in rev 2,
    just recompiled after the label-position fix -- no other visual change
    to module B).
  - THIS SESSION, step 3 rev 4 (user request: shorten B's "command enters
    here" caption to "command"; drop B's "linear case" caption; turn D's
    "execute, observe, replan" caption into three real sub-blocks; make
    every module's OWN internal diagram read strictly left to right, no
    module may fold a chain back on itself).
    PROCESS NOTE: partway through this work, `git log` turned up commit
    8a7e879 ("...step 3 rev 4 -- L-to-R module C, 3-block receding-horizon
    loop"), already on this branch, from a DIFFERENT session
    (Claude-Session session_019dNpHg3wG7w5S5Euh9ZHKg, not this one) --
    someone is running another Cowork/Claude session on this same repo in
    parallel. That commit's modB/modD/kompas_style changes are byte-for-
    byte identical to what this session had independently written for the
    same request (same wording, same coordinates), so no conflict; its
    modC/modD content matched this session's FIRST draft too, before this
    session's own render-and-crop pass (the established verification habit
    for this figure) caught three real bugs that commit did not: (i) an
    arrow in D's three-block loop was drawn to a guessed box half-width and
    visibly cut through the middle of each word instead of stopping at the
    box edge; (ii) C's two new crossing loss labels (L_lin/L_pred) fully
    overlapped each other; (iii) C's reconstruction-branch $D_\psi$ label
    was silently hidden underneath the L_rec loss circle. This entry
    documents this session's work including those three fixes; the base
    geometry decisions (L-to-R row swap, three rhblock nodes, the two
    caption edits) were reached independently by both sessions from the
    same user request, not copied from one to the other. Worth flagging to
    the user: two sessions editing the same figure files concurrently is
    how a commit like 8a7e879 ships without its render ever being crop-
    checked -- there was no second reviewer in that session's loop, only
    in this one.
    Four changes (2 already on the branch via 8a7e879, this session added
    fixes to 2 of them -- see process note above):
      1. modB_koopman.tex: "command enters here" -> "command" (text only).
         Dropped the "linear case: E_phi=D_psi=id" minicap entirely; kept
         both dashed \IdBypass arrows, since each already carries its own
         inline "id" label and does not depend on the dropped sentence to
         be legible.
      2. modD_control.tex: the single caption riding the receding-horizon
         return arrow became three real blocks (new `rhblock` style in
         kompas_style.tex: a plain rounded box, no colour -- G5 reserves
         colour for module identity, not internal captions), connected by
         three short arrows in temporal order (Execute -> Observe ->
         Replan, matching the arrow's own right-to-left-then-up direction).
         This loop is drawn as a loop ON PURPOSE: it is the actual control
         feedback path (execute, observe, replan, repeat), not an
         incidental reversal like the ones fixed in point 4 below, so it
         was NOT flattened to left-to-right.
         BUG caught by rendering, not by the coordinate math: the first
         attempt hand-picked a half-width for each box and placed the
         connecting arrows to stop just outside it: at \scriptsize the
         actual rendered boxes came out wider than guessed, so every arrow
         landed inside its neighbour's box and was clearly visible cutting
         through the middle of the word (checked at 2x zoom on "Observe").
         Fixed by naming each node and re-pointing the arrows at
         `.east`/`.west` node anchors instead of guessed coordinates --
         this lands on the box's true rendered edge regardless of text
         width, and cannot drift out of sync with the font again.
      3. modC_training.tex, the actual "no back-turns" fix: this module's
         own header comment already admitted its lower ("observed") row
         was "folded back and runs right to left" by a prior revision, on
         purpose, so the three loss-link pairs would sit directly above
         one another (short vertical links). The reconstruction branch had
         the same problem on a smaller scale (one right-to-left arrow so
         its result would land back under s_t). Both are now left to right.
         This surfaced a genuine, checked-on-paper-first geometric conflict
         worth recording: the lower row's causal order (s_{t+1} -> Enc ->
         xi_{t+1}, left to right) is the EXACT OPPOSITE of the order needed
         to keep xi_{t+1} under q_t and s_{t+1} under D(q_t) at the same
         time (their upper-row partners sit in that left-right order, not
         swapped) -- so "left to right" and "short aligned loss links"
         cannot both hold once the row direction is fixed. Kept the row's
         x-footprint unchanged (still 2.85-3.97) and just swapped which end
         holds which glyph, which turns L_lin and L_pred from short
         verticals into two ~0.96cm diagonals that cross each other once
         between the rows -- checked against the shared encoder mcE2's own
         box and confirmed neither diagonal crosses it.
         Two more bugs this surfaced, both found by rendering (not by
         re-deriving coordinates -- by now the established pattern for
         every collision this session):
         (a) \symLlin and \symLpred, both placed at the line's own
             pos=0.25 (even after switching to a flat, no-circle label
             style), still overlapped each other almost completely -- the
             two links are near-mirror-symmetric, so any shared pos value
             puts both labels at nearly the same point near the crossing.
             Fixed by abandoning "node on the line" placement for these
             two and hand-placing both in the one pocket that is clear of
             every glyph (below decoder mcD1's bottom edge, above encoder
             mcE2's top edge, between the two column pairs), stacked
             vertically instead of sitting on their own lines.
         (b) the reconstruction branch's own $D_\psi$ label had silently
             vanished from the render. Root cause: moving the reconstruction
             result column to the decoder's right (to fix its own
             right-to-left arrow) meant L_rec's rerouted detour-and-label
             now passed close enough underneath the $D_\psi$ label that the
             loss circle -- drawn afterward, in section 4 -- painted over
             it. Fixed by lowering the detour 0.25cm; re-rendered and
             confirmed the label is back and the circle no longer reaches it.
    Re-ran G2 on all four touched fragments: modB/modD/right_panel all OK,
    0 unmatched; modC still shows its known 7 unmatched (\mathcal{D},
    \mathcal{L}_{rec/lin/pred}, \phi, \psi, \theta) -- unchanged from
    every prior entry, confirmed this is the same pre-existing D19-vs-Q1
    appendix gap, not new drift from this revision's geometry changes.
    Final artefact for review: preview/step_right_panel_v20.png (also
    step_modB_koopman_v14, step_modC_training_v18, step_modD_control_v19
    -- module A untouched, no new preview needed).
  - THIS SESSION, continued (step 3, per user's explicit "先补记账再进入步骤3"):
    wrote right_panel.tex per FIG_PLAN §3.3. Native module canvases read off
    each modX_*.tex \path rectangle: A 2.94x3.03, B 6.53x3.03, C 4.735x3.03,
    D 4.735x3.03. Row content width A+0.25(gap)+B = 9.72 = C+0.25+D -- the two
    rows already matched without any stretching, so panel spacing is uniform
    0.25cm both directions. Placement: A(0,3.28) B(3.19,3.28) C(0,0)
    D(4.985,0). Ran G1 (compile+view, preview/step_right_panel_v1.png),
    G2 (fig_symbol_audit.py right_panel.tex: OK, 0 atoms -- the panel-level
    tags/captions are plain English, not math, so nothing to check against
    02_method.tex), and a targeted G3 check by cropping+2x-zooming the A-B
    seam: the relation-1 arrow (A's s_t bottom-left-aligned column -> B's
    s_t column, both at local y=2.09 in their own modules) lands cleanly on
    B's column border and does not cross either module's own content.
    Two cross-module relations, per FIG_PLAN §3.3:
      1. A's s_t -> B's s_t entry: DRAWN (thinflow arrow, global
         (1.91,5.37)--(3.81,5.37)); both modules' MemColV for s_t happen to
         bottom-align at local y=2.09, so this is a clean level line.
      2. B's y-hat -> D's rollout/plot: NOT drawn. B's y-hat cell sits at
         the bottom-right of B (local (5.64,0.58)); D's mini-axes sits at
         local (2.95-4.45, 0.55-1.65). A literal line between them would
         cross D's own receding-horizon feedback line (local y~2.02,
         x 4.26-4.70) and skim its cost bars (local x 2.95-3.80,
         y 2.12-2.66) -- exactly the collision FIG_PLAN §3.3 names as the
         reason to fall back to a caption. Replaced with a minicap
         "model: from B" inside D's scope, coloured clModB.
      C -> B ("trains B"): also captioned, not drawn (C and B are not
         adjacent at this row spacing; a line would have to route around A
         or B's own box). Minicap "trains B" placed under C's tag,
         coloured clModC.
    Corner tags: \node[ptag,fill=clMod<X>,anchor=north west] in each
    module's reserved top-left strip, text "<letter> <2-word gloss>" (e.g.
    "A Finite memory") -- within the <=4-word budget, module colours used
    ONLY on the tags/captions and not bled into module content (G5).
    PROCESS NOTE (read this before proceeding): steps 2 and 3 were both done
    in this same sitting without a human verdict in between, on the user's
    explicit instruction to close the bookkeeping gap and continue to step 3
    rather than stop and wait after step 2 alone. This is NOT the plan's
    normal cadence (FIG_PLAN §3: one stop-for-review per step) and should
    not be treated as the new default -- the next stop is real: do not draft
    step 4 (master merge into 02_method.tex) until the user has actually
    signed off on step 1 rev2 + step 2 + step 3 together.
  - THIS SESSION: found step 2 (modA/B/C/D) fully drafted and rendered on disk
    (compile + G7 all pass) but uncommitted and un-recorded in this file --
    the plan's own per-step stop-for-review gate had been skipped between step 1
    rev2 and step 2. Ran the G2 symbol audit on all four fragments (results
    above), confirmed G3/G4/G5/G6/G7 by re-viewing the latest PNG of each
    module, rewrote this file to match reality, and is committing the four
    modules + their preview PNGs as a single catch-up commit before asking the
    user to review. No .tex content was changed in this pass.
  - Q1 与 D19 冲突已上报用户，用户裁定图仍按 2x2 四块（C 保留），与主文把训练目标降进
    附录 A1 的处理不同步；若 Phase 2 复议，右栏改 A/B/D 三块是既定备选。
  - 偏离计划 §1.5.1 一处：memory / observation 向量改用单色相明度阶（新宏 MemColV /
    MemRow / LatColVF / LatCol），不用 skill 的四色 tok1..tok4 分类色。理由：一个窗口里
    装的是同一个量的多次取值，分类色会读成四个互不相关的类别。TokColV / TokRow 保留未删。
  - 新增宏（kompas_style.tex 末尾，只加不删）：ScalarCell, ScalarCellT, MatBlock,
    RowSelector, IdBypass, LossLink, FrozenLLM, RulerV, MiniAxes, DimBraceR/T/B,
    Trainable, seqstrip, TokRow, MemColV, MemRow, LatColVF, LatCol.
  - 图元样张 glyph_sheet.tex 不是最终图的一部分，只作步骤 0 的人审件。
  - 步骤 1 rev2（用户 2026-09-19 审后重画）：左栏改成"五个逻辑步骤的控制回路"，
    KOMPAS 在正中。用户指令逐条落实：
    * Frozen LLM -> Inference；Verifier -> Verification；Prompt protocol ->
      Prompt construction（都改成动作名，不是物件名）。
    * KOMPAS 拆成 Modeling 与 Control 两个小 block，左右并列、箭头相连，放在
      KOMPAS 框内部。
    * 删除 offline: learn 虚线框。
    * 左栏公式全部删除。
    * KOMPAS 移到整栏正中（y 中心 3.50 = 画布中心）。
    实现方式：把回路按控制框图的标准顺序旋转成
    Verification -> Memory update -> KOMPAS -> Prompt construction -> Inference,
    再由 Inference 经左侧 response 边回到 Verification。五块 + KOMPAS 居中
    只有在回路恰好五步时才成立。
    rev2 顺带的四个判断（都可否决）：
    (a) 删掉原框 1 "Task \& target"。它是输入不是步骤，且留着就没法让 KOMPAS
        落在正中。目标 r 在右栏模块 D 的绿色目标线上出现。
    (b) "Memory state" -> "Memory update"，把用户的命名规则一并套到这一框。
    (c) offline 框删掉后模块 C（离线训练）在左栏失去锚点，C 色点挪到 Modeling
        下方（训练属于建模那一侧）。
    (d) 左栏 3.4cm -> 3.6cm：Modeling/Control 两个并列小框塞不进 3.4cm。右栏
        由 10.4cm 变 10.2cm，模块 B 按 FIG_PLAN §8 的 A:B = 3:7 备选取 7.14cm，
        仍宽于规格的 7.0cm。
  - 公式移出左栏的副作用：G2 符号审计对 left_loop.tex 变成空跑（0 atoms）。
    全图的记号一致性从此完全由右栏四个模块承担。
  - 步骤 1 rev1 登记的三处偏离 FIG_PLAN §3.1（已被 rev2 取代，存档）：
    (a) 框 3 公式用计划预授权的简写 o_{t+1} ~ p_theta(.|H_t,u_t)（省去 x）；实测全写
        2.83cm，框内可用 2.86cm，留白不够。
    (b) 框 5 标题由 "Memory window" 改 "Memory state"（实测 2.33cm，与右上角色点相撞；
        改后 1.95cm）。措辞与 §3.2 的 memory state 一致。
    (c) 框 6 高 1.02cm（其余 0.72cm），标题拆成 "KOMPAS" + "predict & select" 两行：
        原单行标题实测 3.80cm，比整个左栏还宽。框 6 因此也是全栏唯一的大框，与它是
        本文贡献这件事同向。
    (d) 图元图标只缩进标题行，公式行占满框宽——否则框 3、框 6 的公式一定溢出。
