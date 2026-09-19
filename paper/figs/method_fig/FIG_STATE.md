# FIG_STATE (method figure, TikZ)

updated: 2026-09-19  by: opus session 01QhBcBJhXv13Vc8QT8Rqe2u
step: 0   status: awaiting_user_review
decisions: Q1=C-as-own-module (user, 2026-09-19, 2x2 保持四块)
           Q2=sentence-length (user, 默认)
           Q3=no-bound-glyph, minicap only (user, 默认)
gates: compile=pass  symbol_audit=pass(glyph_sheet.tex 32 atoms, 0 unmatched)
       visual_check=pass(opus viewed preview/step_glyph_sheet_v3.png)
previews_for_review: preview/step_glyph_sheet_v3.png
user_verdict_on_previous: n/a (first step)
next_action: wait for user verdict on the glyph sheet; on approve -> step 1 left_loop
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
