# Alhafni et al. 2024《Personalized Text Generation with Fine-Grained Linguistic Control》：任务/数据集适配性分析（草案 v0.1，2026-08-30）

本文件是继 `LLM_LATENT_STATE_FEASIBILITY.md`（隐状态替代 z_t）、
`SCRIPTED_USER_TURNS_FEASIBILITY.md`（脚本化用户轮次，已尝试后放弃）之后的第三份同类分析记录，
回答的问题是：EACL 2024 Personalize workshop 论文
[Personalized Text Generation with Fine-Grained Linguistic Control](https://arxiv.org/abs/2402.04914)
（代码 [balhafni/personalized-gen](https://github.com/balhafni/personalized-gen)）里的任务是否
适合我们的 Koopman 控制框架，以及能否直接复用它的数据集。**本文件仅做分析记录，不修改
`persona_drift_control/` 已有代码。** `DATA_SOURCES.md` 第4节此前已经把这篇论文的"~50个连续
表层特征"列为候选、"251作者/106k文档数据本身不用"——本文件是对那一行结论的展开核实，结论
方向一致，不推翻。

## 1. 结论摘要

- **任务本身不适合直接套 Koopman**：这篇论文是单次条件生成（一个目标属性区间前缀 + 一句
  prompt → 生成一段续写），不存在多轮/多步的轨迹结构，也没有"每步独立随机施加"的开环激励——
  这两点都是我们协议第1/3节里 Koopman 状态转移辨识（尤其是惯性/衰减）的必要前提，这篇论文的
  任务设计里都不存在。
- **数据集不建议直接用**：底层数据是 Blogs Authorship Corpus、IMDb62、Amazon 5-core
  Reviews 三个静态自然语料（核实自 `data/README.md`），不是对话轨迹，没有人为施加的控制
  输入序列，跟 `DATA_SOURCES.md` 已有判断一致。
- **真正可复用的是特征抽取代码，不是数据集**：`data/utils/annotate_data.py`（POS/依存关系
  频次、FKGL 可读性，另需 `rstfinder` 做 RST 篇章关系）是一套独立于其生成模型的标准 NLP
  抽取流水线，可以直接拿来对我们自己 self-chat 生成的主线回复文本跑一遍，产出约50维连续
  readout，是 `LLM_LATENT_STATE_FEASIBILITY.md`/此前讨论里"免费表层特征"选项的具体落地代码，
  值得记下来供以后用。
- **一个未验证、暂不采纳的旁支想法**：它的"属性区间前缀条件生成"机制，理论上可以反过来改造
  成协议里一个新的、几乎零额外生成成本的输入通道（每轮插入目标属性区间指令，类似 `u_remind`
  的插入方式，用它的抽取脚本读出实际值作为 y_t），但动机（"为什么要控制这些细粒度语言学特征"）
  和可行性都还没验证，记录在案，不建议现在动手。

## 2. 论文任务结构核实（为什么不满足 Koopman 前提）

核实自 [Hugging Face 集成示例](https://github.com/balhafni/personalized-gen#hugging-face-integration)
（`balhafni/personalized-gen` 模型卡里的调用代码）：

```python
ling_atts = {"ADJ": "5-8", "ADP": "10-11", ..., "FKGL": "5-6", "domain": "blog", ...}  # ~50 维
inputs = [''.join([f'{k}:{v}' for k, v in ling_atts.items()]) + prompt]
preds = model.generate(**inputs, ...)  # 一次生成一段续写
```

- **输入**：一次性给定的目标属性区间向量（约50维，覆盖 POS 标签频次、依存关系频次、FKGL
  可读性、`num_sents`/`num_tokens` 等结构特征、`domain` 类别），拼接在 prompt 前作为前缀 token，
  不是逐步施加的信号。
- **输出**：一次生成、一段续写文本；评测是"生成文本的实际属性值是否落在目标区间内"，是单点
  条件生成任务，不是"观察系统状态如何随一系列输入演化"的动力学任务。
- **与我们协议的结构性差异**：我们的 `u_remind` 是每轮独立随机抽取、持续施加在对话历史里的
  开环激励，目的是让 Koopman 的输入矩阵 B 可辨识；这篇论文的"目标属性区间"只在生成前给一次，
  没有"多轮反复施加、观察下一轮/下两轮的滞后响应"这个机制，Q3（是否有惯性）这类问题在这个
  任务设定下根本无法定义——不是"数据不够"，是任务结构本身缺这个维度。

## 3. 数据集核实：三个静态语料，不是轨迹数据

核实自 [`data/README.md`](https://github.com/balhafni/personalized-gen/blob/master/data/README.md)：

> We used the Blogs Authorship Corpus, IMDb62, Amazon 5-core Reviews datasets to construct
> our benchmark.

三者都是**自然产生的静态文本语料**（博客作者语料、IMDb 影评作者归属语料、亚马逊商品评论），
不是任何形式的多轮对话或带控制输入的轨迹记录。即便按"author"分组能拿到同一作者的多篇文档，
这些文档之间也没有实验设计意义上的"每步独立随机控制输入"——顶多是自然产生的时间顺序（比如
博客发帖时间），不满足开环激励辨识的前提，若强行把"author 的多篇文档"包装成"轨迹"，属于
不能自圆其说的构造，跟此前讨论"language_constraints 无探针"方案时担心的"审稿人质疑捏造需求"
是同一类风险，因此不建议尝试。

`DATA_SOURCES.md` 第4节"251作者/106k文档数据本身不用"的判断，核实后确认正确，不需要修改。

## 4. 可复用部分：特征抽取代码（`data/utils/annotate_data.py` 等）

核实自仓库文件列表（`git/trees` API）：

| 文件 | 作用 |
|---|---|
| `data/utils/annotate_data.py` | 主标注脚本，产出约50维语言学特征 |
| `data/utils/fkgl.py` | Flesch-Kincaid Grade Level 可读性计算 |
| `data/rstfinder/` | 独立的 RST 篇章关系解析器（第三方库，`elaboration`/`attribution`/
  `contrast` 等篇章关系特征依赖它） |

这套流水线是**独立于论文生成模型的纯 NLP 标注代码**，输入任意文本、输出约50维连续特征，不
需要下载他们的数据或模型。可以直接对我们 `selfchat.py` 生成的 `agent_message` 主线回复文本跑
一遍，作为 `y_probe` 旁边的免费连续 readout（呼应 `DATA_SOURCES.md` 第4节、
`LLM_LATENT_STATE_FEASIBILITY.md` 里提到的思路）。

**需要注意的成本细节**：这不是严格零成本——POS/依存句法解析本身较轻，但 `rstfinder` 是一个
独立的、依赖预训练模型文件的老代码库（RST 篇章解析），引入它比单纯的 formality/sentiment
打分要重一些工程量；如果只要 POS/依存/可读性这部分（不要 RST 篇章关系），可以只用
`annotate_data.py` 里对应的那部分逻辑，跳过 `rstfinder` 依赖，成本会更低。这部分细节留到
真正决定要用的时候再细看代码，本文件不展开。

## 5. 未验证的旁支想法：反过来把"属性区间前缀"改造成新输入通道

**仅作记录，不建议现在采纳**：论文的"目标属性区间前缀 → 条件生成"机制，理论上可以反过来
改造成协议里的第四个输入通道——每轮在用户消息前插入一段"目标语言学属性区间"指令（工程上和
`u_remind` 的插入方式一样，不需要额外生成），再用 `annotate_data.py` 读出主线回复的实际属性值
作为 y_t。如果成立，这会是一个几乎零额外生成成本的 (u_t, y_t) 通道，比现在依赖探针分叉的
`y_probe` 便宜得多。

但这个想法目前只是看完代码后的推测，至少有两件事没做：
1. **动机没想清楚**：为什么要控制"依存关系频次""FKGL 区间"这类细粒度语言学特征，对应什么
   现实场景/研究问题，还没有论证，直接做容易被质疑"因为代码现成才做"而不是"因为这是个真问题"。
2. **可行性没验证**：这类细粒度语言学目标能不能通过简单的 prompt 插入有效控制（不像
   `u_remind` 重申人格设定那样直觉上应该有效），完全没有先验，需要先做类似
   `DATA_COLLECTION_PROTOCOL.md` 第7节那种小规模信号探针才能知道。

## 6. 与本项目现有讨论的关系

- 呼应此前"是否有其他成本合理的适用任务"讨论里的方案1（隐状态/免费表层特征替代探针）：
  本文件确认了 Alhafni 特征抽取代码是这个方向里"免费表层特征"选项的具体、可直接复用的实现，
  不再只是文档里提到的一个名字。
- 与方案2（language_constraints 无探针，因动机不足被否决）、`SCRIPTED_USER_TURNS_FEASIBILITY.md`
  （scripted-user，因脚本质量检验不过被放弃）共享同一条经验教训：凡是"因为现成代码/数据便宜
  就想用"的方向，都需要先把动机和可行性讲清楚，否则容易在下一步被质疑捏造需求或者直接测不出
  预期效果。第5节的旁支想法目前就卡在这一步，暂不采纳。
