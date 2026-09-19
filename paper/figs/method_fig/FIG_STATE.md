# FIG_STATE (method figure, TikZ)

updated: 2026-09-19  by: opus session 01QhBcBJhXv13Vc8QT8Rqe2u
step: 1   status: awaiting_user_review
decisions: Q1=C-as-own-module (user, 2026-09-19, 2x2 保持四块)
           Q2=sentence-length (user, 默认)
           Q3=no-bound-glyph, minicap only (user, 默认)
gates: compile=pass
       symbol_audit=pass(left_loop.tex 21 atoms / glyph_sheet.tex 29 atoms, 0 unmatched)
       visual_check=pass(opus viewed step_glyph_sheet_v3.png and step_left_loop_v5.png)
       G3 no overlap / arrows on borders / min \scriptsize = pass
       G4 <=4 words per node, no sentences, no quoted examples = pass
       G5 only box 6 carries the `ours` border; module colours only in the dots = pass
       G6 no numbers in the figure = pass
previews_for_review: preview/step_glyph_sheet_v3.png (step 0),
                     preview/step_left_loop_v5.png (step 1)
user_verdict_on_previous: n/a (first session; step 0 and step 1 submitted together
                          per FIG_PLAN \S9 item 3)
next_action: wait for user verdict on step 0 + step 1; on approve -> step 2,
             four module fragments (Sonnet x4 in parallel, Opus reviews and merges)
notes:
  - Q1 与 D19 冲突已上报用户，用户裁定图仍按 2x2 四块（C 保留），与主文把训练目标降进
    附录 A1 的处理不同步；若 Phase 2 复议，右栏改 A/B/D 三块是既定备选。
  - 偏离计划 §1.5.1 一处：memory / observation 向量改用单色相明度阶（新宏 MemColV /
    MemRow / LatColVF / LatCol），不用 skill 的四色 tok1..tok4 分类色。理由：一个窗口里
    装的是同一个量的多次取值，分类色会读成四个互不相关的类别。TokColV / TokRow 保留未删。
  - 新增宏（kompas_style.tex 末尾，只加不删）：ScalarCell, ScalarCellT, MatBlock,
    RowSelector, IdBypass, LossLink, FrozenLLM, RulerV, MiniAxes, DimBraceR/T/B,
    Trainable, seqstrip, TokRow, MemColV, MemRow, LatColVF, LatCol.
  - 图元样张 glyph_sheet.tex 不是最终图的一部分，只作步骤 0 的人审件。
  - 步骤 1 登记的三处偏离 FIG_PLAN §3.1：
    (a) 框 3 公式用计划预授权的简写 o_{t+1} ~ p_theta(.|H_t,u_t)（省去 x）；实测全写
        2.83cm，框内可用 2.86cm，留白不够。
    (b) 框 5 标题由 "Memory window" 改 "Memory state"（实测 2.33cm，与右上角色点相撞；
        改后 1.95cm）。措辞与 §3.2 的 memory state 一致。
    (c) 框 6 高 1.02cm（其余 0.72cm），标题拆成 "KOMPAS" + "predict & select" 两行：
        原单行标题实测 3.80cm，比整个左栏还宽。框 6 因此也是全栏唯一的大框，与它是
        本文贡献这件事同向。
    (d) 图元图标只缩进标题行，公式行占满框宽——否则框 3、框 6 的公式一定溢出。
