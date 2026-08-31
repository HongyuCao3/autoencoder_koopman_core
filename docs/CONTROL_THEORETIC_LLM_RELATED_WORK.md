# 控制论方法做 LLM 控制：2025–2026 相关工作调研（草案 v0.1，2026-08-31）

调研动机：避免本项目在套用 Koopman 时出现过多自创设计、偏离已有学术脉络。本文件回答三个问题：
(1) 2025–2026 顶会是否存在用控制论模型做 LLM 控制的工作；(2) 它们如何建模（状态、时间轴、
动力学、控制器）；(3) 是否存在与 Koopman 类似的、对**惯性**或**线性性**有明确需求的建模。
**仅调研记录，不修改代码与协议。** 主论文缺 Related Work 一节，本文件可直接作为其素材来源。

## 1. 总览：按"时间轴"分类

控制论建模的第一个选择是"t 是什么"。已有工作分三个时间轴，加一支奠基性理论：

| 工作 | 发表 | 时间轴 | 状态 x_t | 动力学模型 | 控制器 | 线性性需求 | 惯性需求 |
|---|---|---|---|---|---|---|---|
| A-LQR（Skifstad, Yang, Chou，Georgia Tech，arXiv 2604.19018，2026-04） | arXiv（投稿中） | **层** k | 末 token 各层残差流激活 | 逐层 Jacobian 线性化 → LTV 系统 δz_{k+1}≈A_k δz_k + B_k δu_k | LQR（Riccati 递推，跟踪对比方向设定点） | **核心假设且实证验证**："层间动力学局部线性近似良好" | 内建（层间状态传递即惯性） |
| PID Steering（Nguyen et al.，**ICLR 2026**，arXiv 2510.04309） | ICLR 2026 | 层 k | 各层激活 | 层间误差动力学 ē(k+1)=Ā(k)ē(k)−Ā(k)u(k)+w(k)（Jacobian 局部线性化） | PID（证明 ActAdd 等现有 steering 方法都只是 P 控制器；PI 消稳态误差） | 局部线性化 + 有界 Jacobian | 内建 |
| LiSeCo（Cheng, Baroni, Amo Alonso，arXiv 2405.15454） | arXiv/OpenReview | token t | 各层激活 | 不建转移模型；线性 probe 定义"安全区域" | 每步最小范数投影回安全半空间（带闭式保证） | 线性 probe（线性表示假设） | 无（逐 token 静态约束） |
| CBF-LLM（Miyaoka & Inoue，arXiv 2408.15625；后续 2511.03121，2025-11） | arXiv/OpenReview | token t | 输出文本嵌入 | 不显式辨识；CBF 条件作用于逐 token 转移 | 控制屏障函数安全滤波器（对候选 token 重排/否决） | 弱 | 弱（逐步不变集） |
| BarrierSteer（Tran, Verma, ... Rus, Xiao，NUS+MIT，arXiv 2602.20102，2026-02） | arXiv（投稿中） | token t | 解码步隐状态（layer 20） | 隐状态随机转移；一步内有限差分近似（局部 Lipschitz） | 学出的神经 CBF + 闭式 steering 解 | 一步内局部线性/Lipschitz | 弱（一步近似） |
| PID 推理调控（Bharadwaj，arXiv 2506.18831） | arXiv | token（24-token 块） | 层 20 隐状态池化 + 冗余概率分类器 | 无显式模型 | 标准 PID 调 steering 系数 α_t（误差 = 冗余概率 − 目标） | 隐含（α 与效果近似线性耦合，未论证） | 隐含 |
| **NBF-LLM**（Hu, Robey, Liu，CMU，**ICML 2025**，arXiv 2503.00187） | **ICML 2025** | **轮** k | 对话嵌入状态（m=768） | **学出的**离散时间状态转移 f_θ + 观测 g_θ（3 层 MLP，在对话轨迹上 MSE 训练） | 神经屏障函数安全滤波器（逐轮过滤对抗 query） | 无（非线性 NN 动力学） | **核心前提**：显式承认非 Markov/记忆，多轮攻击轨迹渐进漂向不安全区 |
| **Drift No More**（Dongre et al.，UIUC+Adobe，arXiv 2510.07777，2025-10） | arXiv/OpenReview | **轮** t | 逐轮 KL 散度（测试模型 vs 目标一致参考模型） | **带恢复力的有界随机过程**（均值回复递归 + 可控干预项） | 提醒干预（实验验证降低散度，符合理论预测） | 标量线性递归 | **核心对象**（但结论是稳定平衡而非失控漂移） |
| GenCtrl（Cheng, Amo Alonso 等，Apple，arXiv 2601.05637，2026-01） | OpenReview 投稿 | 轮 | 对话文本 | 无结构模型；Monte Carlo + PAC 界估计可达/可控集 | 无（分析，非综合） | 无 | 无显式 |
| Magic Word（Bhargava et al.，arXiv 2310.04444，2023–24） | arXiv（奠基） | token | 上下文 | 自注意力的可达集理论（k-ε 可控性） | prompt 作为控制输入 | 无 | 理论层面 |
| When is Your LLM Steerable?（Fan et al.，UMD，arXiv 2606.11599，2026-06） | arXiv | （剂量轴） | — | steering 强度 α 的剂量-响应：倒 U 形，UNDERSTEER/SUCC/OVERSTEER | — | **反面证据**：α-效果关系非单调 | — |

另：Koopman×Transformer 的检索结果（DeepKoopFormer 等）全是"用 Koopman 结构改造时间序列
预测网络"，与"控制 LLM 行为"无关；**未发现任何已发表的"turn 级 Koopman/EDMD 辨识 + 可控性
结构分析"工作——本项目的 gap 仍然成立**。

## 2. 对三个调研问题的回答

**(1) 领域存在且在升温。** 层轴上 ICLR 2026 正式接收了 PID Steering；轮轴上 ICML 2025 正式
接收了 NBF-LLM；CBF、LQR、可达性分析各有一支。用控制论语言做 LLM 控制不是偏离学术范畴，
而是一个正在形成的可引用脉络——本项目的 Related Work 可以直接挂进去。

**(2) 建模方式高度分化，分化的轴就是时间轴。** 层轴工作把一次前向当一条轨迹（惯性免费、
线性性可实证）；token 轴工作把一次生成当轨迹（多为安全滤波，动力学建模很浅）；轮轴工作
（NBF-LLM、Drift No More、GenCtrl）与本项目同轴，是直接竞争/对照对象。

**(3) 与 Koopman 需求最同构的三篇：**
- **A-LQR**：对"线性性"的需求和用法与我们几乎一致——不假设全局线性，而是实证验证局部
  线性近似够好，然后享受线性最优控制的全部工具箱。区别只在时间轴（层 vs 轮）。它是"线性
  surrogate 这条路在 LLM 上走得通"的最强旁证。
- **NBF-LLM**：对"惯性"的需求与我们一致——它的全部前提就是多轮对话状态有记忆、攻击轨迹
  是渐进漂移的，并且它成功地在轨迹数据上学出了 turn 级转移函数（ICML 2025 接收，证明
  turn 级动力学建模可发表、可实现）。区别在它用黑盒 MLP，不做结构分析。
- **Drift No More**：把"漂移"本身建成带恢复力的均值回复过程——直觉上就是我们上次讨论的
  x_{t+1}=a·x_t+… 里 a<1 的情形：状态有延续（a≠0，惯性存在），但自发趋势被拉回平衡点
  而不是失控发散。

## 3. 对本项目的五点含义

1. **我们的空结果有了外部理论呼应。** Drift No More 在不同任务/模型上得到与我们 pilot 同
   方向的结论：良性设定下漂移是"稳定的、噪声受限的平衡"，不是失控退化；且提醒干预在他们
   那里**有效**（我们的 u_remind 无效应，差异可能在模型/干预构造，值得对比其实现细节）。
   这支持把控制问题改述为"平衡点调节/扰动抑制"（与 STOLFO 分析第 4 节一致），且改述后
   有文献可引，不算自创叙事。
2. **惯性确实存在的任务域已经被指认：对抗性多轮攻击。** Crescendo/ActorAttack 这类攻击
   之所以成立、NBF-LLM 之所以能学出有用的转移函数，都因为对抗压力下的对话状态是渐进演化
   的。良性 self-chat 漂不动（我们的 pilot + Drift No More），对抗性压力下漂得动（NBF-LLM
   的 plant）。**任务选型上这是比"换 readout"更根本的信号：给系统加一个持续的对抗/诱导
   扰动源，惯性可能就"出现"了。**
3. **gap 确认，delta 清晰。** 轮轴上已有的三家：GenCtrl 无结构模型（我们的既定对标）、
   NBF-LLM 黑盒 NN 动力学 + 滤波（不做可控性/谱分析）、Drift No More 标量现象学递归
   （无输入矩阵辨识、无 MIMO）。"提升线性 surrogate + 可控性矩阵/Gramian/谱的结构分析"
   这个位置仍然空着。
4. **设计纪律有了参照系。** 文献里每个组件都有已发表的版本：执行器（steering 向量，
   ICLR 2025/2026 多篇）、turn 级嵌入状态（NBF-LLM）、线性 surrogate 的辩护方式（A-LQR 的
   "先实证验证局部线性再用 LQR"）、平衡点叙事（Drift No More）。组装这些不算过度设计；
   反过来，凡是文献里没有对应物的自创机制（比如过于复杂的多通道激励设计）都应更审慎。
5. **一个警示。** When is Your LLM Steerable? 报告 steering 强度-效果是倒 U 形、成功率
   仅 10–23%——线性 B 矩阵假设在 α 大时必然失效，协议里 steering 电平应保守取在剂量-响应
   的近线性段（先做 α 扫描，正呼应 STOLFO 分析第 8 节第 2 步）。

## 4. 参考文献

- Skifstad, Yang, Chou. Local Linearity of LLMs Enables Activation Steering via Model-Based
  Linear Optimal Control. arXiv:2604.19018, 2026.
- Nguyen, Vu, Pham, Zhang, Nguyen. Activation Steering with a Feedback Controller. ICLR 2026.
  arXiv:2510.04309.
- Hu, Robey, Liu. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking
  Attacks. ICML 2025. arXiv:2503.00187. 代码 github.com/HanjiangHu/NBF-LLM.
- Dongre, Rossi, Lai, Yoon, Hakkani-Tür, Bui. Drift No More? Context Equilibria in Multi-Turn
  LLM Interactions. arXiv:2510.07777, 2025.
- Cheng, Baroni, Amo Alonso. Linearly Controlled Language Generation with Performative
  Guarantees. arXiv:2405.15454.
- Miyaoka, Inoue. CBF-LLM: Safe Control for LLM Alignment. arXiv:2408.15625；Control Barrier
  Function for Aligning Large Language Models. arXiv:2511.03121.
- Tran, Verma, Wong, Low, Rus, Xiao. BarrierSteer: LLM Safety via Learning Barrier Steering.
  arXiv:2602.20102, 2026.
- Bharadwaj. Adaptive Activation Steering for Efficient LLM Reasoning via Closed-Loop PID
  Control. arXiv:2506.18831.
- Fan, Cheng, Li, Feizi, Zhou. When is Your LLM Steerable? arXiv:2606.11599, 2026.
- Bhargava, Witkowski, Shah, Thomson. What's the Magic Word? A Control Theory of LLM
  Prompting. arXiv:2310.04444.
- Cheng, Amo Alonso 等. GenCtrl. arXiv:2601.05637, 2026（本项目既定对标）.
