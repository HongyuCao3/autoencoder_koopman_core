# FIG_STATE (method figure, TikZ)

updated: 2026-09-19  by: opus session (bookkeeping catch-up + step 3 + step-3 revision, this session)
step: 3 (rev 2)   status: awaiting_user_review
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
                     preview/step_modB_koopman_v12.png,
                     preview/step_modC_training_v14.png,
                     preview/step_modD_control_v17.png,
                     preview/step_right_panel_v12.png (step 3 rev 2, CURRENT
                       -- supersedes v1: fixed the neuron-overlap bug in
                       every scaled net and 4 separate text-overlap bugs,
                       see the note above)
user_verdict_on_previous: not yet recorded in this file for step 1 rev2 or step 2 --
  rev2's changes already encode detailed user feedback from a prior review (see
  notes below), but no explicit approved/revise verdict was logged before the
  same session went on to draft all four step-2 modules without commit/state
  update. This entry is a bookkeeping catch-up: it records the true on-disk
  state (step 2 fully drafted, gates run, nothing committed) so a fresh session
  is not misled by the stale step:1 entry that preceded it. All 5 images above
  were just sent to the user for review together.
next_action: step 3 (right_panel.tex) has now been drafted and rendered in
             the same sitting as the step-2 catch-up, ALSO without waiting for
             a user verdict first -- see note below, this is a second instance
             of the same process gap, done knowingly this time with the user's
             explicit go-ahead (see chat: "先补记账再进入步骤3"). On user
             approve of step 1 rev2 + step 2 (four modules) + step 3 (panel)
             together -> step 4 (master assembly, wire into 02_method.tex,
             render the real page). On revise -> iterate the named
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
