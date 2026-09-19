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
                     preview/step_left_loop_v6.png (step 1, rev 2)
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
