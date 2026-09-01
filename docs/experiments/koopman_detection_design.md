# 设计笔记：让 Koopman 代理模型具备"检测"能力（而不只是控制）

和 [koopman_defense_pilot.md](koopman_defense_pilot.md) 同一类"供跨会话接续"的记录。
**新开一次对话想知道"给 Koopman 加检测能力这条线想过哪些方案、做到哪一步了"，看这份文档。**

## 背景与问题设定

`koopman_defense_pilot.md` 记录的 Phase A→E 已经闭环：`KoopmanMPCController` 用拟合出的
`richer_abs_sign`（`nu=1, mu=2`）代理模型做前瞻规划，选择是否插入安全提醒。这条线里
**"检测"这件事已经存在，但不是 Koopman 做的**——`y_safety`（LLM-judge 对当前轮回复的打分）
本身就是"这一轮是否已经被攻击侵蚀"的检测信号，`ThresholdController` 直接读它做反应式判断。

用户提出的问题是：既然控制器能"自动判断何时加防御性提示"，那背后应该也对"攻击的出现"有所
感应——Koopman 代理模型能不能把这种感应显式做出来，而不只是隐含在选动作的过程里？

区分两个不同层次很重要：
- **内容层检测**（"这一轮回复本身是否已经不安全"）——`safety_judge.py` 已经做了，Koopman
  模型看不到攻击文本，帮不上这个忙。
- **动力学层检测**（"当前这条轨迹的演化方式是否符合模型学到的攻击动力学，还是出现了模型没
  见过的模式"）——这是 Koopman 代理模型原则上能提供、但目前完全没有利用的额外信息。

下面记录讨论过的四个方案，按接入现有代码的难度排序。

## 方案 1：预测残差 / innovation 信号（成本最低，只需复用现有 `step`/`readout`）

`modeling/evaluate.py::one_step_error` 已经在批量评测里算这个量，只是聚合成了一个 MSE 标量。
在线场景下把它按轮次展开：每轮用 `t-1` 时刻的真实状态 `z_{t-1}` 算
`surrogate.readout(surrogate.step(z_{t-1}, v_{t-1}))`，得到模型对这一轮安全分的预期，
和真实观测到的 `y_probe[t]` 相减，就是控制理论里标准的 innovation/新息。不需要新数据、不需要
改 `fit()`。**已执行，见下方"方案 1 执行结果"。**

## 方案 2：把 MPC 内部的前瞻预测显式暴露成"预警"输出

`KoopmanMPCController._simulate` 已经在做多步滚动预测，但预测轨迹本身在选完动作后就被丢弃了。
可以加一个旁路方法（如 `predict_horizon(z, horizon) -> list[float]`），把未来几轮的预测安全分
序列记录下来（比如写入 `y_safety_forecast_min` 这一列）。当预测的未来最低点即将跌破阈值、
而当前观测值还没跌破时，就是比 `ThresholdController` 更早一步的预警——价值在于利用了
`nu=1, mu=2` 状态里编码的惯性（Phase C 已确认这个惯性在有防御的情况下依然显著存在）。
**未执行**：需要先看方案 1 的残差信号是否有意义，再判断这个"提前量"是否值得做。

## 方案 3：双 regime 对比（检测意义最强，但需要新数据）

现在拟合数据全部来自攻击轨迹，模型学到的是"攻击压力下安全分怎么演化"，**没有"正常对话下
安全分应该怎么演化"这个基线**，原则上无法回答"这条轨迹是不是在被攻击，还是只是正常波动"。
仓库里 `drift_confirmation_pilot`/`pressure_screening` 虽然 schema 一致（`y_probe`/
`u_remind`），但测的是人格漂移分而不是安全分、任务域也不同，**不能直接当良性基线用**。真要
做，需要照抄 `attack_trajectory.py` 的框架、把 `attack_bank` 换成正常多轮对话，用同一个
`safety_judge` 打分，采一批"良性 + 安全打分"轨迹，拟合第二个 `KoopmanSurrogate`，在线对同一段
观测算两个模型的似然/残差比值。**未执行**，是这四个方案里唯一需要新采集数据的。

## 方案 4：状态里塞入攻击特征（改 lifting，不改核心框架）

`_psi(z) = [z, extra_features_fn(z)]` 这个接口本来就是为非线性提升留的（`abs_sign_extra_features`
是现成例子）。可以写一个新的 `extra_features_fn`，除了 `abs(y)/sign(y)`，再拼入轮次编号，或者
攻击文本相对 `attack_bank` 已知攻击原型的相似度，让状态本身携带"当前对话像不像已知攻击模式"的
证据。改动量最小（一个新函数），但引入文本特征后需要重新验证 `controllability_diagnostics`
这类诊断是否还站得住。**已执行，见下方"方案 4 执行结果"——负面结果。**

---

## 方案 4 执行结果

**实现方式（和文档原计划有一处偏差，如实记录）**：写完才发现 `extra_features_fn` 的签名
`Callable[[np.ndarray], np.ndarray]` 只看得到数值化的 `z`，看不到轮次原文，没法在这个接口
内部算文本相似度。真正落地的做法是把 `dataset.py::ReducedStateConfig` 加一个可选
`aux_cols: tuple[str, ...] = ()` 字段（默认空，不影响任何现有调用），让 `z_t` 变成
`[y 历史, v 历史, aux_1(t), ...]`——辅助列作为"当前轮"取值直接拼进 `z`（不带自己的滞后阶数），
`z_next` 里对应取下一轮的值，和 `y`/`v` 走同一套 NaN 跳过与状态推进逻辑。这样文本特征就能通过
预处理阶段算好、以数值列的形式进入 `z`，`extra_features_fn`（`no_extra_features`/
`abs_sign_extra_features`）本身不用改一行。

**相似度特征本身**：新增 `modeling/content_similarity.py`，纯 numpy 从零实现的 TF-IDF +
余弦相似度（不引入 sklearn，理由和 `controllability_diagnostics` 当年不跨包 import 一致）。
参照语料只取"这次拟合的训练集里的攻击"（`attack_id` 排除掉 8 个 held-out），避免 held-out
攻击的原文直接躺在自己的参照语料里造成平凡意义上的"重合"。`scripts/fit_koopman_defense_model.py`
新增 `--aux-cols attack_similarity`：用 train_rows 建参照语料、给 train/held-out 都打分，
再把参照语料原文存进 fit report 的 `content_reference_texts` 字段；
`scripts/fit_koopman_benign_model.py` 读这个字段复用同一个冻结语料，给良性文本
（`question` 列）打同样的相似度分；`scripts/evaluate_koopman_detector.py` 评测时对 Phase E/F
的 eval 行做同样的标注。三个脚本都加了 `--aux-cols`，默认空、不影响原有跑法。

**拟合质量：加入这个维度让攻击 regime 的 rollout MSE 变差**（`nu=1,mu=2` 基线 ARX 0.051→0.064，
richer 0.043→0.074），良性 regime 基本不受影响（0.027→0.028）。说明"当前轮"的攻击相似度
不是能被安全分历史线性外推预测的量——这符合预期：文本内容和安全分数值走的是两条不同的因果链，
模型被迫用同一套线性 A/B 去同时预测两者，多花的自由度换来的是拟合变差，不是变好。

**检测准确率：仍然是负面结果，没有清楚跑赢 always-attack 基线**。在 4 种
`{arx, richer_abs_sign}` 攻击/良性模型组合里都测了一遍（避免只挑一个看起来最好的组合汇报）：

| 攻击模型 | 良性模型 | 逐轮准确率 | 逐轨迹准确率 |
|---|---|---|---|
| arx | arx | 0.517–0.546（两版语料） | 0.463–0.575 |
| arx | richer_abs_sign | 0.540–0.574 | 0.513 |
| richer_abs_sign | arx | 0.472–0.477 | 0.438–0.488 |
| richer_abs_sign | richer_abs_sign | 0.466–0.506 | 0.450–0.500 |

（区间来自两版参照语料：第一版纯 bag-of-words，第二版加了一份手写停用词表后重跑——诊断见下段。
8 个组合里最好的是 arx/richer_abs_sign，逐轮 0.574；`always-attack` 基线是逐轮 0.727、
逐轨迹 0.800，8 个组合没有一个够到。）**比方案 3 的纯残差比较（0.551/0.475）略好但幅度很小，
没有改变"跑不赢平凡基线"这个结论。**

**诊断了原因，不是实现 bug**：直接算了一下 held-out 攻击轮次和良性轮次相对参照语料的相似度分布——
攻击轮次均值 0.315(std 0.091)，良性轮次均值 0.286(std 0.102)，两组几乎完全重叠（加停用词表前是
0.338 vs 0.311，加了之后差距几乎没变，说明不是停用词污染的问题）。**根因是 bag-of-words 相似度
在这批短指令式文本上分辨力太弱**：攻击查询和 MT-Bench 良性问题都大量使用"explain/write/how
would you/describe"这类指令式措辞，这些词在只有 ~20 个攻击的参照语料里出现频率也不算低、
IDF 压不干净，真正带区分度的内容词（"phishing" vs "poem"之类）在整句余弦相似度里权重被稀释了。
参照语料本身也小（22 个攻击、每个 4-6 轮），统计上撑不起更细的词表。

**结论：方案 4 在这个最小实现（TF-IDF 词袋 + 冻结小语料）下也是负结果**，且诊断指向的瓶颈
（短指令式文本上词袋相似度分辨力不足）不是这次实现的偶然缺陷,是这类特征在这个数据规模下的
真实天花板——继续往前推需要换成语义 embedding 相似度或扩大参照语料规模,而不是在当前 TF-IDF
实现上继续调参（比如换 n-gram、调 IDF 平滑）。四个方案目前都跑完了：方案 1（残差）负面、
方案 2（前瞻预警）因方案 1 负结果未执行、方案 3（双 regime）负面、方案 4（内容特征）负面。
按最上面"这个负结果说明了什么"一节的框架看,这不是"Koopman 方法不行",而是这套项目当前能拿到的
状态表示（少量数值读数 + 小规模参照语料）信息量上限如此;真要把检测这条线做起来,需要的是更大规模
的数据或更强的文本表示,不是在现有框架内换一种拟合方式。

产物：`outputs/koopman_detection_content_feature/attack_fit_report.json`（攻击 regime，
含 `content_reference_texts`）、`benign_fit_report.json`（良性 regime）、
`two_regime_{attack_model}_{benign_model}/two_regime_detector_report.json`（4 种组合 ×
2 版语料的检测器评测结果）。代码改动：`modeling/dataset.py`（`ReducedStateConfig.aux_cols`）、
新增 `modeling/content_similarity.py`，三个脚本加 `--aux-cols`/`--content-reference-report`
选项，均向后兼容（默认空，不影响任何已有调用），新增单测覆盖 aux_cols 状态构造和相似度计算
（`tests/test_modeling_dataset.py`、`tests/test_content_similarity.py`，`pytest` 全绿）。

---

## 方案 1 执行结果

脚本：`scripts/analyze_koopman_innovation.py`（纯 CPU，几秒）。加载 Phase C/D 已经拟合好的
`richer_abs_sign`（`nu=1, mu=2`）模型（`outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json`），
在 Phase E 的四臂闭环数据（`zero_control`/`constant_remind`/`threshold`/`koopman_mpc`，各
8 个 held-out 攻击 × 2 seed）上逐轮做**一步预测**（每步都用真实观测到的 `z_t` 重新起算，不是
自由滚动）：`residual = y_true[t+1] - readout(step(z_t, v_t))`。

结果落盘：`outputs/koopman_detection_innovation/innovation_residuals.csv`（逐轮明细）、
`innovation_summary.json`（汇总）。

| 臂 | n | mean residual | std | mean\|residual\| | persistence MAE（"下一轮=上一轮"基线） |
|---|---|---|---|---|---|
| zero_control | 32 | +0.073 | 0.353 | 0.307 | 0.219 |
| constant_remind | 32 | -0.226 | 0.203 | 0.233 | 0.211 |
| threshold | 32 | +0.176 | 0.262 | 0.273 | 0.258 |
| koopman_mpc | 32 | +0.138 | 0.238 | 0.221 | 0.172 |

- **residual 与"这一轮实际降幅"（`y_true - y_prev`）的相关系数 = 0.526**（中等正相关；注意
  两者共享 `y_true`，所以这个相关系数天然偏正，不能直接读成"模型能预判剧烈下跌"）。
- **侵蚀穿越阈值事件**（`y_prev >= 0.7` 且 `y_true < 0.7`，128 轮里 19 轮）：这些轮次的平均
  `|residual|` = 0.309，其余轮次 = 0.250——方向上支持"侵蚀发生时模型更容易被打脸"，但差距
  不大（约 24%），不是干净的分离。

**诚实结论（负面/持平结果，如实记录）**：这个一步预测残差**没有清楚跑赢"下一轮=上一轮"这个
零成本基线**——除了 `koopman_mpc` 臂（0.221 vs 0.172，模型明显更差）之外，其余三臂的
`mean|residual|` 和 `persistence_mae` 相当接近，`threshold` 臂甚至只差一点。也就是说，
**用当前这个模型的一步预测误差直接当"异常/攻击存在"信号，现在还立不住**——它比"什么都不用、
直接看上一轮读数有没有变"提供的信息量有限。

**可能的原因，值得记录**：`richer_abs_sign` 是在 Phase B 的 `random_excite`（p=0.5，独立
Bernoulli 插入）策略下拟合的，而 Phase E 四臂的插入策略分别是"从不"（0%）、"总是"（100%）、
阈值反馈、MPC 反馈——四种都是训练时从未见过的输入分布（`u_remind` 序列的统计结构系统性不同）。
`mean residual` 的正负号和幅度按臂系统性不同（`constant_remind` 明显偏负，其余偏正）支持这个
解释：**残差里混着"策略分布外"和"攻击动力学异常"两种信号，没有分离**，这也是为什么直接拿它
当检测器不够干净。方案 3（良性基线对比）不会有这个混淆问题，因为要对比的是"同一策略下攻击 vs
良性"，而不是"这一策略下和拟合时的策略是否一致"。

**下一步建议**：方案 1 目前的结果不支持继续投入方案 2（在一个混杂着策略分布外效应的残差上加
"预警阈值"，噪声太大意义有限）。如果要继续做检测这条线，方案 3（良性基线双 regime 对比）
是更值得投入的方向——但需要先决定是否值得为此新采一批良性对话数据。

## 方案 3 执行结果

**意外发现：不需要新采数据。** `koopman_defense_pilot.md` 的 Phase F（良性 helpfulness 代价
检查，2026-09-01 完成）为了衡量控制器对良性流量的误伤，本来就把 MT-Bench 良性对话跑过了
和攻击轨迹完全同一套 `safety_judge`（用固定占位 `harmful_goal` 打分）——`trajectories.jsonl`
里已经有 `y_safety`/`y_probe`/`u_remind`，schema 和攻击域一模一样。写方案 3 的第一版设计时
还以为要专门去采，实际检查 Phase F 输出后发现数据已经现成，直接省了这一步。

**第一步：拟合"良性 regime"模型**（`scripts/fit_koopman_benign_model.py`，纯 CPU，几秒）。
合并 Phase F 四臂（`zero_control`/`constant_remind`/`threshold`/`koopman_mpc`）的
`trajectories.jsonl` 作为一个辨识数据集（换取 `u_remind` 输入多样性）——**踩了一个坑**：
Phase F 四臂复用同一批良性会话内容，`trajectory_id` 在四臂之间完全相同（例如
`mtbench_coding__seed0` 在四个臂里都叫这个名字），直接合并会被 `group_by_trajectory` 静默
合并成一条错误的长轨迹；脚本里在加载时把 `trajectory_id` 加上臂前缀（`f"{arm}__{tid}"`）
处理掉了这个问题。按 `benign_id`（8 个 MT-Bench 类别）切 75/25，held-out 类别
`mtbench_stem`/`mtbench_writing`。`nu=1,mu=2` 下 ARX（`held_out_rollout_mse=0.027`）比
`richer_abs_sign`（0.037）更好——和攻击域相反（那边非线性提升更好）——符合直觉：良性对话下
`y_safety` 基本贴着 1.0 小幅波动，线性 ARX 就够，用不上 `abs_sign` 这种为不对称侵蚀反应设计
的提升。良性模型选 ARX。

**第二步：双 regime 残差比较检测器**（`scripts/evaluate_koopman_detector.py`）。评测集严格
避开两个模型各自的训练数据：攻击标签用 Phase E 四臂（8 个 held-out 攻击 × 2 seed × 4 臂）、
良性标签用 Phase F 四臂里 held-out 的 2 个类别 × 2 seed × 4 臂。每轮/每条轨迹分别用攻击模型
和良性模型算一步预测残差，`|residual|` 更小的模型判给哪个 regime。

| 粒度 | n | 准确率 | "always 猜 attack"基线 |
|---|---|---|---|
| 逐轮 | 176（128 attack / 48 benign） | **0.551** | 0.727 |
| 逐轨迹 | 80（64 attack / 16 benign） | **0.475** | 0.800 |

**诚实结论（负面结果）：这版检测器比"什么都不判断、无脑猜 attack"这个平凡基线还差**，两个
粒度上都是——大部分攻击轨迹被误判成良性（逐轮 74/128，逐轨迹 41/64）。

**排查了两个可能原因，都不是主因**：
1. **不是量纲/噪声尺度不匹配**——两个模型各自训练集上的 `train_one_step_mse` 很接近
   （攻击 `richer_abs_sign`=0.059，良性 `arx`=0.050），不存在"一个模型残差天生数值更大"这种
   会系统性偏向另一个模型的量纲问题。
2. **不是策略分布外效应的混淆**——只看 `zero_control` 臂（Phase E/F 里两边都是"完全不插提醒"
   的匹配策略，排除了方案 1 发现的那种策略不一致污染）单独算，逐轮准确率 0.568，仍然低于
   该子集的 always-attack 基线（0.727）。

**真正的原因,如实记录**：很多攻击轨迹——尤其是被 `constant_remind`/`threshold`/`koopman_mpc`
成功防住的那些——`y_safety` 全程维持在较高水平，动力学形状（`nu=1,mu=2` 状态里能看到的东西：
最近的安全分读数 + 最近两轮插没插提醒）和良性对话几乎没有区别；良性模型在这些"看起来很稳"的
攻击轮次上预测得反而更准。**这不是实现细节的 bug，是这条检测思路本身的天花板**：仅凭
`y_safety` 数值和提醒插入历史这两类信号,防御生效的攻击轨迹在动力学层面就是长得像良性对话——
要把它们分开,需要方案 4 那种能看到"这轮在聊什么"的内容特征,而不是在方案 3 现有的状态表示上
继续调（比如加方差归一化）。

**结论：方案 3 到此完整跑通一轮（数据→拟合→检测器→评测），是清楚的负结果，不建议在当前状态
表示上继续投入调参。如果还要做检测这条线,下一步应该直接跳到方案 4（状态里塞入攻击文本相似度
之类的内容特征），而不是在方案 3 的残差比较规则上打补丁。**

产物：`outputs/koopman_detection_benign_baseline/koopman_fit_report.json`（良性模型）、
`outputs/koopman_detection_two_regime/two_regime_residuals.csv` + `two_regime_detector_report.json`
（逐轮残差明细 + 混淆矩阵/准确率汇总）。

## 这个负结果说明了什么：不是"Koopman 不行"，是这套状态表示的信息不够

方案 3 跑完之后容易被误读成"Koopman/线性代理模型这套方法有局限"，需要在文档里说清楚**不是
这样**，避免以后重新捡起这条线时被这个误读带偏。

**根本原因是信息层面的，不是模型家族层面的。** `nu=1, mu=2` 状态里只装了"最近的安全分读数"
和"最近两轮插没插提醒"这几个数字，这点信息本身就不够分辨"被防御压住的攻击"和"正常对话"——
换成 LSTM、随机森林、任何模型家族去学同一组输入，结果不会变好，因为问题出在喂给模型的原材料
不够，不在模型怎么处理这些原材料。两处已有证据支持这个判断：Phase C 里非线性提升只比纯 ARX
略好、这次良性模型甚至是 ARX 更好——说明加大模型的非线性容量没有换来更多分辨力,如果输入里
真有区分度,哪怕很弱的线性模型也该能部分利用上。

**更值得记录的是一组干净的对照：控制够用 ≠ 检测够用。** Phase E 已经证明同一个 `nu=1,mu=2`
状态表示拿来做控制（选择要不要插提醒）是够用的——`koopman_mpc` 打赢了三个基线；这次方案 3
证明同一个状态表示拿来做检测（分辨这是不是攻击）是不够用的。这两者不矛盾：**控制只需要知道
"现在插提醒会不会让安全分变好"这个方向性信息（对应 B 矩阵），完全不需要知道"这条轨迹是不是
攻击"**——控制器可以在从不识别攻击的情况下,纯靠"低分就压一下"的反馈把安全分控制住;而**检测
需要的是轨迹的身份判断，是一个分类问题，需要的信息量和控制不是一回事**。所以 Phase E 的成功
和方案 3 的失败是同一个状态表示在两个不同任务上呈现不同充分性的对照实验,不是自相矛盾。

**一个和这套方法实际用法相关、值得记的真实局限**：攻击轨迹很短（4-6 轮），Phase C 已经发现
`mu=3` 直接数值退化撑不住，状态阶数被迫压得很低（`state_dim=3`）。这是"轨迹太短→状态阶数
上不去"的具体约束，跟数据规模有关，不是 Koopman 理论本身的天花板，但在这个项目的实际条件下
确实限制了状态能装多少历史信息——以后往方案 4 加内容特征时，也得在这个很紧的阶数预算里塞，
是个实打实的工程约束，不是可以靠"选更好的模型"绕开的问题。
