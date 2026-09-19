# FIG_STATE (method figure, TikZ)

updated: 2026-09-19  by: opus session (bookkeeping catch-up, this session)
step: 2   status: awaiting_user_review
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
                     preview/step_modA_memory_v5.png,
                     preview/step_modB_koopman_v10.png,
                     preview/step_modC_training_v10.png,
                     preview/step_modD_control_v8.png
user_verdict_on_previous: not yet recorded in this file for step 1 rev2 or step 2 --
  rev2's changes already encode detailed user feedback from a prior review (see
  notes below), but no explicit approved/revise verdict was logged before the
  same session went on to draft all four step-2 modules without commit/state
  update. This entry is a bookkeeping catch-up: it records the true on-disk
  state (step 2 fully drafted, gates run, nothing committed) so a fresh session
  is not misled by the stale step:1 entry that preceded it. All 5 images above
  were just sent to the user for review together.
next_action: on user approve of step 1 rev2 + step 2 (four modules) -> step 3
             right_panel.tex (Opus assembles, per FIG_PLAN \S3.3); on revise ->
             iterate the named module only, bump version, re-run gates.
notes:
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
