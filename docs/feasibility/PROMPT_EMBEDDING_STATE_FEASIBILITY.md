# 用独立文本 embedding 模型编码 prompt（或 safety_prompt+prompt）替换/增广 Koopman 状态：简记（草案，2026-09-02）

**未开始实现，仅记录想法与已有证据，供之后决定是否继续。** 背景：`experiments/ae_baseline_plan.md`
的 AE(encoder-decoder) baseline 跑完后，讨论"隐空间建模的对象是否可以替换"——当前
`AEKoopmanSurrogate`/`KoopmanSurrogate` 的 `z_t` 都是标量读数历史
（`[y_safety 历史, u_remind 历史]`），提案是换成/加入一个独立文本 embedding 模型编码后的
prompt（攻击者当轮文本，或 safety system prompt + prompt 拼接）。

## 和已有两份文档的关系

这个提案落在两条已有调查线中间，不是全新方向：

1. **[LLM_LATENT_STATE_FEASIBILITY.md](LLM_LATENT_STATE_FEASIBILITY.md)**：分析过换成
   Qwen3-4B **自己的内部隐状态**（2560 维，需要本地权重访问）。结论"可行但高维小样本过拟合
   风险大，建议先 pilot 验证再决定，暂不作为默认协议"——从未实现。本提案的关键区别：用**独立
   的文本 embedding 模型**（不是被控 LLM 自己的内部表示），维度可控（通常几百维），对闭源/API
   模型也适用，不继承该文档"仅适用于本地可访问权重模型"这条限制（第 8 节）。
2. **[../experiments/koopman_detection_design.md](../experiments/koopman_detection_design.md)
   方案 4**：已经真正实现并跑过"把攻击 prompt 文本相似度塞进 z_t"——`modeling/content_similarity.py`
   的 TF-IDF 词袋余弦相似度，通过已有的 `ReducedStateConfig.aux_cols` 机制接入。**负结果**：
   加入后 rollout MSE 变差，诊断出根因是"这批短指令式文本上词袋相似度分辨力不足"（held-out
   攻击轮相似度均值 0.315 vs 良性轮 0.286，几乎完全重叠），文档明确写了"继续推进需要换语义
   embedding，不是调 TF-IDF 参数"，但因参照语料只有约 22 个攻击，判断为低优先级，未真正换成
   embedding 重跑。本提案就是这条被搁置的"下一步"，只是应用场景从"检测特征"扩展到"AE baseline
   的 z 空间输入"。

## 主要取舍（讨论时提到的，未验证）

- **维度**：真实 sentence embedding（如几百维）远小于 Qwen 内部隐状态的 2560 维，但仍远超
  当前 `state_dim=3`，直接喂给 AE encoder 大概率复现 `LLM_LATENT_STATE_FEASIBILITY.md` 第 5
  节的高维小样本过拟合问题——需要先降维（PCA / 岭回归投影）或者只取一个标量特征（如最近邻
  相似度，方案 4 的做法）而不是整段 embedding 向量。
- **"safety prompt + prompt" 拼接的稀释问题**：如果 safety system prompt 在轮次间基本不变，
  编码整段拼接文本会被恒定前缀稀释掉真正变化的攻击者文本信号——和方案 4 诊断出的"指令式套话
  稀释内容词"是同一类问题的变体。更合理的做法可能是只编码当轮变化的部分（攻击者文本），或者
  编码"当轮 - 上一轮"的某种差分表示。
- **语义 embedding 能否解决方案 4 的"分辨力不足"是未验证的假设**：小语料本身的统计天花板
  （约 22 个攻击、每个 4-6 轮）可能仍然限制效果，换用更强的文本表示不保证翻盘，需要实证。
- **验证目标和方案 4 不完全一样**：方案 4 测的是"能否提升攻击/良性二分类检测准确率"；这次的
  提案测的是"能否提升代理模型的 one-step/rollout 预测质量"（`ae_baseline_plan.md`/
  `koopman_defense_pilot.md` 的评测口径）——两者用同一个特征来源，但目标指标不同，结论不能
  直接互相替代，即使方向一致也需要分别验证。

## 如果要继续：建议的最小验证路径（未执行）

沿用方案 4 已经验证过的工程路径（`ReducedStateConfig.aux_cols` + `fit_koopman_*.py` 的
`--aux-cols`），把 `content_similarity.py` 的 TF-IDF 相似度换成一个轻量开源 sentence-embedding
模型的余弦相似度（结构改动最小，直接复用已有 pipeline），跑一遍 `ae_baseline_plan.md` 同一套
one-step/rollout 对比，看数字有没有变化——比重新设计整个状态空间成本低得多。**暂不实施，等待
后续决定。**
