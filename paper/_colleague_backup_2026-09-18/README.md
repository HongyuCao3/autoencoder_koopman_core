# 同事 LaTeX 模板原样备份（2026-09-18）

本目录是 `paper/iclr2027_koopman_alignment/` 在 2026-09-18 的**逐字节副本**，不做任何修改。
备份的目的：接下来按 `paper/WRITING_PLAN_2026-09-18.md` 改造 Method 章、重建 `contract.yaml`、
拆分 `sections/` 时，随时可以和同事原稿逐行对照，不必回 git 历史里翻。

## 里面有什么

| 文件 | 内容 | 后续用途 |
|---|---|---|
| `3-method.tex` | 同事写完的 Method 全章（8 个公式、记号体系、figs/method.png 的图注） | Phase 2 的改造底稿；全篇记号以它为准 |
| `4-experiment.tex` | 主表：IFBench / IFEval / COLLIE 上 Qwen3-4B 与 Qwen3-8B × {Base, Koopman, TMPC, RE-Control} 共 **88 个数字** | 已逐条抄进 `paper/evidence/numbers_benchmark.yaml`（`n_bench_001`–`n_bench_088`），Phase 3b 的 Table 1 由脚本从 yaml 生成 |
| `Main.tex` | 同事的主文件与章节 `\input` 顺序 | 对照新 `paper/main.tex` 的结构差异 |
| `1-intro.tex` / `2-related.tex` / `5-con.tex` / `6-appendix.tex` | 空壳，各一行 `\section{...}` | 无 |
| `figs/method.png` | 方法图（四阶段） | Phase 2 复制到 `paper/figs/method.png` 后引用 |
| `iclr2027_conference.{sty,bst,bib}`, `math_commands.tex`, `fancyhdr.sty`, `natbib.sty` | ICLR 2027 模板文件 | 已在 `paper/` 根目录各存一份 |

## 数据口径提醒

`4-experiment.tex` 的 88 个数字**没有随附原始日志、seed、置信区间或配置**。
用户 2026-09-18 裁决（写作计划 D3）：暂按可靠处理，正文用点估计口径，并在 Phase 3b
明写"数字为点估计，区间待补"。一旦同事补齐来源，需同步更新
`numbers_benchmark.yaml` 的 `ci` 与 `supersede_reason` 字段。

**本目录只读。** 任何改造写到 `paper/sections/` 和 `paper/main.tex`，不要改这里。
