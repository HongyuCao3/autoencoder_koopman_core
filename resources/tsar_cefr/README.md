# `tsar_cefr` 资源与来源（Phase 1 可得性核查，2026-09-13）

**数据未 vendoring**，只记溯源。是否落盘进仓库待用户裁决（下面「许可」一节）。
线的定义见 [`../../docs/NAMING.md`](../../docs/NAMING.md)，死亡条件见
[`../../docs/experiments/tsar_cefr_kill_criterion.md`](../../docs/experiments/tsar_cefr_kill_criterion.md)。

## 可得性：全部公开，无 gating

| 物件 | HuggingFace / GitHub id | revision (sha) | license |
|---|---|---|---|
| trial 数据 | `cardiffnlp/TSAR2025_SharedTask_RCTS_Trial-Data` → `tsar2025_trial.jsonl` | `c4d590b66dbb5dbd31c8a158f068c0adcfbe0af3` | **未声明** |
| test 数据 | `cardiffnlp/TSAR2025_SharedTask_RCTS_Test-Data` → `tsar2025_test.jsonl` | `3f4a9d38f82eb9126c2a069f0bd009301fb3df28` | **未声明** |
| CEFR 分类器 1 | `AbdullahBarayan/ModernBERT-base-doc_en-Cefr` | `3c29f5fbcdc753e99bb437ff9303df983486915b` | apache-2.0 |
| CEFR 分类器 2 | `AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr` | `b00d1d4780e46f6410ea8a8509649044dee18298` | apache-2.0 |
| CEFR 分类器 3 | `AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2` | `83337437aa82277e96b293665dc3186088a4a839` | apache-2.0 |
| 保义读出 | `davebulaval/MeaningBERT` | `f40ff1edfdf9f6a6121eb5701b8557426a268495` | — |
| 官方评测脚本 | `tsar-workshop/tsar-2025-shared-task` → `code/tsar2025_evaluation_script.py` | 待 pin | 待查 |

三个分类器的标签集**一致**：`A1 A2 B1 B2 C1 C2`（6 级），与计划 §3.1 的 $\ell=\sum_{k=1}^{6} k\,p_k$ 假设相符。

**许可**：两份数据的 HF 卡片**没有 license 字段**，仓库既有先例（`resources/sequor/`、
`resources/ergo_gsm8k_sharded.jsonl`）都是带明确许可才 vendoring。**建议不 vendoring**，
改为运行时按上表 sha 拉取并把 sha 记进 `hydra_run_config.json`——溯源强度相同，不承担再分发。

## 实测规模与分布（在 scratchpad 下载核对，未入库）

| | 行数 | 不同源文本 | 目标级 | reference |
|---|---|---|---|---|
| trial | 40 | **20** | `a2` 20 / `b1` 20 | 每行 1 条 |
| test | 200 | **100** | `A2` 100 / `B1` 100 | 每行 1 条 |

字段：`dataset_id` / `text_id` / `original` / `target_cefr` / `reference`。
`text_id` 形如 `01-a2` / `21-b1`——**源文本 id 与目标级拼成**，同一源文本的两个目标级是两行。

源文本长度（词）：trial 最短 36 / 中位 71 / 最长 115；test 最短 46 / 中位 87 / 最长 149。
→ `max_new_tokens` 按「段落长度 ×1.5」≈ 300 token 可覆盖最长段落。

**大小写不一致**：trial 的 `target_cefr` 是小写（`a2`/`b1`），test 是大写（`A2`/`B1`）。读入时归一化。

## 与计划 §3.1 的四处出入

1. **test 是 100 个源文本，不是 80。** 计划 §5.3 与 §6.1 按 80 算的 N 与 GPU-2 规模要上修 25%。
   GPU-1 的「100 源文本」与 test 的源文本数恰好相符。
2. **reference 不是「两名标注者」，是每 (源文本, 目标级) 一条。** trial 的 40 条参考 = 20 源 × 2 目标级。
   G-T1 判据①「20 条源文本与其 40 条参考简化」数量对得上，但构成不同，判据措辞要按 (源, 目标级) 配对重写。
3. **目标级恰好是 {A2, B1}**，与计划一致；官方任务定义里的 A1 在这两份数据里不出现。
4. **官方 CEFR 评测器是三个分类器的集成**，取 top-1 置信度最高的那个分类器的 argmax
   （`max((top1(d1), top1(d2), top1(d3)), key=score)`），不是单个分类器、也不是概率期望。

## 官方评测脚本口径（已读源码）

- CEFR 合规：weighted F1、adjacent accuracy（|Δ| ≤ 1）、**RMSE**（在整数编码的 CEFR 级上）。
- MeaningBERT：分别对 `original` 与 `reference` 各算一次，除以 100 归一。
- BERTScore F1：对 `reference`。
- **脚本里没有加权总分**，各指标分开报；排名另用 `tsar2025_autorank.ipynb`。

⚠️ 计划 §5.4 的主量「TSAR 官方加权分（RMSE 50% + MeaningBERT 源 16.7% + 参考 33.3%）」
**在官方脚本里不存在**。主量定义缺官方依据，待用户裁决。
