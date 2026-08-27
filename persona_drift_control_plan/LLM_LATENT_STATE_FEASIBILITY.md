# 用 Qwen3-4B 内部隐状态替代/增广工程特征作为 Koopman 状态 z_t：可行性分析（草案 v0.1，2026-08-27）

本文件是 `DATA_COLLECTION_PROTOCOL.md`（通道 A/B：提醒强度、注意力增益）与 `KV_INJECTION_MONITORING.md`（通道 D：KV 注入）之外的一条**备选/并行方向**的可行性记录，不改变现有输入通道设计，评估的是：把 Koopman/AE 状态 `z_t` 从当前的标量遵循分 `y_probe`（及 `y_formality`、`y_sentiment`）替换或增广为 Qwen3-4B 生成过程中的内部隐状态向量，是否可行、是否值得做。

本文件仅做分析记录，**不修改** `autoencoder_koopman_core` 已有的 `core.py`、`configs/`、`datasets/`，也不修改 `persona_drift_control/` 已有代码；提出的所有改动点均以"新增文件/新增方法/新增列"的形式设计，供后续决定是否实施时参考。

## 1. 结论摘要

- 技术上可行：agent 与模拟用户都用 `persona_drift/chat_model.py` 里的 `AutoModelForCausalLM.from_pretrained` 本地加载 Qwen3-4B，权重完全可访问，拿隐状态不需要额外基础设施或换模型部署方式。
- 对 `autoencoder_koopman_core` 核心代码零侵入：`AugmentedKoopmanModel` / `DeepAugmentedKoopmanAutoencoder` 只要求 `Z_t` 是数值矩阵，state_dim 对它们透明，接入方式是新增 dataset/state config，`core.py` 不需要改。
- 主要成本不在"能不能拿到"，而在：①隐状态维度（2560）远超现有 latent_dim（16–64）导致的高维小样本过拟合风险；②turn 级池化规则本身是未经验证的建模假设；③浮点向量的存储体积与仓库现有"repo-local、GitHub 友好"的策略有冲突；④与已有 KV 注入监控协议在"读取模型内部信号"这件事上有重叠，工程上应合并而非另起一套前向逻辑。
- 建议：不作为默认协议，先做小规模 pilot（呼应 `DATA_COLLECTION_PROTOCOL.md` 第 7 节"采集前信号探针"的做法）验证隐状态是否真的比现有标量 readout 更好拟合，再决定要不要转正。

## 2. 背景：现状 vs 提案

| | 现状（`DATA_COLLECTION_PROTOCOL.md`） | 本文件提案 |
|---|---|---|
| z_t 来源 | 事后测量：探针打分 `y_probe`（确定性 Python 打分函数）、`y_formality`（calibrated scorer）、`y_sentiment`（Cardiff RoBERTa） | 生成过程中的模型内部表示：某层 hidden state 的池化向量 |
| 维度 | 标量或个位数向量 | 数百至数千维（Qwen3-4B 单层 hidden_size = 2560） |
| 是否需要额外前向 | 探针需要在分叉副本上单独生成+打分（K=4 次重复） | 若只取主线回复对应 token 的隐状态，可复用 `generate()` 已经算过的前向，不需要额外前向 |
| 可解释性 | 高（分数含义明确） | 低（需要额外验证隐状态与遵循分的关系） |
| 与 core.py 接口 | 已验证：`normalized_output`/`effective_norm` 等标量列直接可用 | 需新增列/侧车文件，`core.py` 本身不变 |

## 3. Qwen3-4B 的可抽取信号（已核实规格）

`chat_model.py` 用的是 `AutoModelForCausalLM.from_pretrained(model_id)`（HF transformers backend），对应 [`Qwen/Qwen3-4B` 的 `config.json`](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)：

| 字段 | 值 |
|---|---|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads`（GQA） | 8 |
| `intermediate_size` | 9728 |

因为模型是本地加载、非 API 调用，前向传播时传 `output_hidden_states=True`（或 `output_attentions=True`）即可拿到全部 37 层（含 embedding 层输出）的隐状态张量，不需要改模型加载方式，也不需要 hook 之外的额外基础设施。这一点和 `KV_INJECTION_MONITORING.md` 第 3 节里读取注意力权重 `a_bank`/`a_sys` 用的是同一类前向内省能力，本质上是同一个"读模型内部信号"的基础设施，只是探针不同（hidden_states vs attention weights）。

## 4. 抽取设计选项

- **层选择**：只取最后一层（语义最"任务对齐"，但可能过度依赖具体输出 token）；取中间层（如第 18 层，语义更抽象、可能更贴近"人格状态"这类持续性表示）；或同时保留 2–3 层（早/中/晚）做对比，不建议一开始就多层拼接（见第 5 节维度问题）。
- **token 位置/池化**：
  - 回复最后一个生成 token 的隐状态（近似"总结状态"，与 KV 注入监控里"槽位不占对话位置、持续被读"的直觉一致）；
  - 回复全部 token 的均值池化（更平滑，但混入了措辞细节）；
  - 探针问答对应 token 的隐状态（如果想让 z_t 与 `y_probe` 的语义对齐更紧，但这样就必须在探针分叉副本上取值，等于每轮多做一次可复用的前向）。
- **与现有采集流程对齐**：`DATA_COLLECTION_PROTOCOL.md` 已经在主线生成之外做探针分叉；如果只要主线回复最后 token 的隐状态，可以在 `chat_model.py` 的 `generate()` 内部顺带拿到（`generate(..., return_dict_in_generate=True, output_hidden_states=True)`），不需要新协议；如果要对**已采集完的历史数据**（`raw_generation` 文本）离线补抽，则需要用同一 `model_id`/精度重新跑一遍前向，补抽结果与"生成时刻真实隐状态"存在细微差异（采样解码路径不同、KV cache 状态不同），需要在文档里明确声明这一限制，不能等价视为原始信号。

## 5. 维度与建模影响

- 单层 2560 维远超 `core.py` 当前 `latent_dim`（默认 16，示例配置常见 16–64）与 `hidden_dim`（默认 64）的量级。直接套用现有 `DeepAugmentedKoopmanAutoencoder` 的 MLP 尺寸会明显欠拟合；调大网络又会在当前数据规模（`DATA_COLLECTION_PROTOCOL.md` 首期规模：320 条轨迹 × 16 轮 = 5,120 个 turn 级样本）下过拟合。
- 若同时拼接多层（如 3 层 × 2560 = 7,680 维），样本/维度比进一步恶化，不建议作为首期方案。
- 更现实的路径是先做线性降维（PCA 或有监督的岭回归投影到 O(50–200) 维）再喂给现有 encoder，而不是让 encoder 直接吃 2560 维原始向量。
- "是否要做成真正的 VAE（概率化 latent，而不是现有的确定性 AE）"与"z_t 是否来自 LLM 内部表示"是两个独立问题：即使换成隐状态输入，`DeepAugmentedKoopmanAutoencoder` 仍可以保持确定性；如果确实想要概率化 latent，更需要先降维——当前几千条样本规模下 KL 项基本不可靠估计。

## 6. 工程改动点（均为增量，不改动现有内容）

- `persona_drift/chat_model.py`：新增一个方法（例如 `generate_with_hidden_state()`），返回文本 + 选定层/位置的隐状态向量；不修改现有 `generate()` 的签名与行为，`run_signal_screening.py` 等现有调用方不受影响。
- 数据表：不建议把 2560 维浮点数组塞进现有 JSONL 列（体积会爆炸：仅单层、5,120 行、float32 估算约 52 MB，若保留 3 层约 157 MB），应存成侧车文件（`.npz`/`.parquet`），JSONL 里只留一个引用/索引列。这与 README.md 里"当前目录约 71 MB，最大单文件约 31.7 MiB，无需 Git LFS"的现状策略有直接冲突，需要提前决定：侧车文件是否入库、是否需要切到 Git LFS 或仅存 `/scratch`。
- `autoencoder_koopman_core` 侧：新增 `configs/dataset/<qwen_hidden_state_xxx>.yaml`，必要时新增一个新的 state family（如果要把隐状态和现有 delay embedding 结合）。`core.py` 的训练/推理/checkpoint/诊断逻辑不需要改，符合 `CODE_DESIGN.md` "Extension Points" 里描述的扩展方式。

## 7. 建议的验证顺序（pilot 优先，不建议直接上全量规模）

呼应 `DATA_COLLECTION_PROTOCOL.md` 第 7 节"采集前的 1 小时信号探针"，提出三个对应问题，建议用小规模 pilot（如 5 条 prompt × 2 seed × 16 轮）先回答：

1. **隐状态里是否有 `y_probe` 的可读信息**：用 pilot 数据做岭回归，从池化隐状态回归 `y_probe`，对比一个 naive baseline（如回复长度、关键词计数）的 R²，隐状态明显更优才有继续做下去的意义。
2. **隐状态的逐轮转移是否比标量 `y_probe` 更适合被 Koopman/AE 拟合**：用现有 `AugmentedKoopmanModel`/`DeepAugmentedKoopmanAutoencoder` 分别在标量状态和（降维后的）隐状态上做一步预测，对比 one-step MSE。
3. **抽取与存储的边际成本是否可接受**：实测单条轨迹多拿一份隐状态的额外前向时延（若只取主线最后 token 应接近零开销）、磁盘占用，判断是否值得从"侧车文件"方案转成正式协议的一部分。

三项里任一明显不达标（隐状态不比标量更可预测、或成本远超收益），都应停在 pilot 阶段，不进入正式采集。

## 8. 风险与局限

- 仅适用于本地可访问权重的模型；若未来把 agent 换成闭源 API 模型，这条路线连同 `KV_INJECTION_MONITORING.md` 里"仅适用于可访问缓存的开源模型"的限制会同时失效。
- 隐状态是强上下文依赖的逐 token 表示，"每轮一个池化向量"是否构成一个"够用的"Markov/delay-embedded 状态，没有先验保证，是需要第 7 节 pilot 验证的经验问题，不能假设成立。
- 高维 + 小样本的过拟合风险是研究性工作，不是接一个 dataset config 就能自动解决的；需要显式的降维/正则化策略并做消融。
- 大浮点向量的存储/版本管理与仓库现有"repo-local、GitHub 友好、不用 Git LFS"的设计目标冲突，需要在实施前就定下侧车文件的存放策略。

## 9. 与现有两份协议的关系

- 不替代 `DATA_COLLECTION_PROTOCOL.md` 的通道 A/B 设计，可以在同一次采集轨迹上顺带多存一份隐状态，理论上不需要重新设计激励协议。
- 与 `KV_INJECTION_MONITORING.md` 里的 `a_bank`/`a_sys`（注意力权重）同属"读取模型内部信号"，工程上应该合并成一次前向里同时拿 `output_hidden_states` 和 `output_attentions`，而不是为隐状态另起一套单独的前向逻辑。

## 10. 建议

暂不作为默认协议。等主协议（通道 A/B）与 KV 注入监控（通道 D）都验证出"标量 `y_probe` 空间上的 Koopman 有效"之后，再考虑把本文件方案作为"状态空间增广"的下一步实验；目前它的收益（更丰富的状态表示）是否覆盖其成本（高维、存储、过拟合风险、与现有仓库策略的冲突）还没有实证支持，不建议在验证之前投入正式采集规模的工程量。
