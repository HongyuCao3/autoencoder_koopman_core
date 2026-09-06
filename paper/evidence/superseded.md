# Step 2b：作废/修正关系裁决

执行者：Opus 5，2026-09-06。输入：`paper/evidence/numbers.yaml`（Step 2a 抽出的 533 条）
＋ `ABLATION_STUDY.md`、`docs/experiments/*.md`、`docs/README.md`，以及三处直接核对的产物
JSON 与源码。产物：本文件，并已回填 `numbers.yaml` 的 `status` / `superseded_by` /
`supersede_reason`。

**出口闸门**：`status == unknown` 的条目 **0** 条；每条 `superseded` 都有 `superseded_by`
指向一条非 superseded 的条目。裁决分布：`current` 244 / `caveated` 172 / `superseded` 117。

---

## 一、对 Step 2a schema 的两处修订（Step 4 必须知道）

### 1.1 `status` 从两值改成三值

计划里 `status` 隐含只有 `current` / `superseded` 两种。裁到一半就撞上第三类事实：
**唯一可得的测量，但产出它的协议后来被本项目自己判定为被混淆，而且没有重跑过**。
例如 `ABLATION_STUDY.md` 第一阶段的状态定义消融全部跑在 `epochs=200` 下，第八阶段证明这个
预算对 2/8 任务不够，但第一阶段从未在早停下复核。把它记成 `current` 等于隐瞒；记成
`superseded` 则没有 `superseded_by` 可指，直接违反闸门。

因此新增第三个值：

| status | 含义 | Step 4 能否引用 |
|---|---|---|
| `current` | 可直接引用 | 可以 |
| `superseded` | 同一个量有更好的估计，`superseded_by` 指向它 | **不可以**（只能在显式的"修正前/修正后"对照里出现） |
| `caveated` | 唯一可得，但协议已知被混淆；或只能作为某个对照的一侧出现 | **可以，但必须在同一句话里说出 `supersede_reason` 里的限定条件** |

`claim_ledger` 的规则相应从"evidence 必须 `status: current`"改成"不得引用 `superseded`；
引用 `caveated` 时必须把限定条件写进该条 claim 的措辞里"。172 条 `caveated` 里包含这条线
最核心的几个数字（惯性前提、Phase J 全部臂级读数），一刀切排除会把论文掏空。

### 1.2 `superseded_by` 允许指向 `caveated`

计划要求指向一条 `current`。做不到，而且不该做：Phase H 被 Phase I 取代，但 Phase I 的
安全分同样是自评 judge 打的（E3），它自己就是 `caveated`。强行把 Phase I 标成 `current`
是用一个隐瞒换一个闸门。改成"`superseded_by` 指向一条**非 superseded** 的条目"——闸门的
真实意图（不留悬空链、不指向已作废值）完全保住。已机械校验：0 条违规。

---

## 二、事件清单

计划点名了五次改写事件。逐条裁决时确认了这五次，另外查出**五次计划没有列出的**（E6–E10），
其中 E7 是本次新发现、此前任何文档都没有记录过的错误。

### E1 · `modeling/dataset.py` 的 "v 对齐" 时序错位（触发：2026-09-02）

`build_reduced_state_pairs` 把 `v` 与 `z_t`（已含 `y_t`）配同一个 `t`，学到的是提醒的
**残留效应**而非同轮直接效应。污染范围是**一切拟合出来的量**，以及**任何消费了这种拟合的
控制器**。

- **superseded（53 条）**：`n_base_001`、`n_base_007`–`n_base_017`、`n_def_010`–`n_def_013`、
  `n_def_019`、`n_def_080`–`n_def_095`、`n_def_097`、`n_op_018`、`n_op_020`、`n_op_022`、
  `n_op_024`、`n_op_026`、`n_op_028`、`n_op_030`、`n_op_032`、`n_op_034`–`n_op_041`、
  `n_op_043`、`n_op_046`、`n_op_048`
- **caveated（41 条）**：`n_def_014`–`n_def_018`（Gramian/秩，从未重算）、
  `n_def_024/028/032/037/038/041/045/048/052/053/056`（Phase E/F 的 koopman_mpc 臂，
  控制器吃的是错对齐的拟合，未重跑）、`n_def_068`–`n_def_079`（旧对齐离线重放）、
  `n_op_049`–`n_op_059`（检测线方案 3/4 的旧对齐拟合）、`n_op_066/067`
- **修正后引用**：代理模型层一律用 v-aligned 值——`richer_abs_sign` = `n_base_018`
  (0.0684)、`arx` = `n_base_019` (0.0683)；闭环用 Phase I（`n_def_124`–`n_def_127`）；
  检测线用 `n_op_019/021/023/025`（方案 1）与 `n_op_042/044/045/047`（方案 3）。
- **理由**：修正后 `B` 符号翻正（−0.0589 → +0.0209），交互项方向从"状态越差越不该提醒"
  翻成正确方向（corr(margin, y_probe) +1.0 → −0.9993），Phase H 的两条具名失败轨迹被救回。
- **最高风险的一条**：`0.043`（`n_def_019`/`n_base_008`）是全仓库被交叉引用最多的过期值，
  在 `lstm_baseline_plan`、`ae_baseline_plan`、`koopman_detection_design`、
  `koopman_phaseI_policy_closed_form` 四份文档里都作为对照基线出现。真值是 0.0684。

### E2 · 早停混淆（触发：2026-09-03，`ABLATION_STUDY.md` 第八阶段）

`epochs=200` 是人工选的、从未验证是否收敛，第八阶段证明对至少 2/8 任务不够
（`sentiment_t5` 重建 loss 差 100 倍、`average_word_length_t5` 差 67 倍）。

- **superseded（8 条）**：`n_core_004`、`n_core_015`、`n_core_017`、`n_core_019`、
  `n_core_021`、`n_core_023`、`n_core_025`、`n_core_027` → 分别指向
  `n_core_049`、`n_core_045`、`n_core_046`、`n_core_047`、`n_core_048`、`n_core_050`、
  `n_core_051`、`n_core_052`
- **caveated（28 条）**：`n_core_001`–`n_core_003`、`n_core_005/006`、`n_core_009`–`n_core_014`、
  `n_core_029`–`n_core_041`、`n_core_043`（第一/三/四a/六/七阶段，全部跑在 `epochs=200`
  且从未重跑），以及重复记录 `n_base_049/050`、`n_op_065`
- **理由**：两个旗舰效应量都被改写——`average_word_length_t5` 从"AE 赢 44.6%"直接反转成
  打平（AE 反输 0.4%），`sentiment_t5` 从"线性赢 78.1%"收窄到 8.1%。方向 6/8 不变，量级不保。
- **一条判断，写清楚**：线性 baseline 侧（`AugmentedKoopmanModel`）是闭式岭回归、无训练轮数，
  E2 碰不到它，所以全部 11 条线性对照值判 `current`；第八阶段的表格本身也在复用它们。
- **另一条判断**：`n_core_053/055`（早停前的重建 loss）判 `current` 而非 superseded——
  它们是第八阶段用来**证明**未收敛的证据，是这次混淆的测量者，不是受害者。

### E3 · 自评 judge → 独立 judge（触发：2026-09-03，job 15519620）

防御线从 Phase A 到 J 的每一个 judge 分都是 `judge_model == agent_model == Qwen3-4B` 打的。
1605 行离线重打分（agent 文本直接复用，配对是构造性精确的）。

- **superseded（8 条）**：`n_def_133/138/142/146/150/154/158`（同时受 E5）、`n_read_086`
- **caveated（77 条）**：`n_def_001`–`n_def_006`、`n_def_008/009`、`n_def_021`–`n_def_032`、
  `n_def_042`–`n_def_045`、`n_def_049`、`n_def_057`–`n_def_059`、`n_def_063`、`n_def_065`、
  `n_def_124/126`、`n_def_159`–`n_def_164`、`n_def_176`–`n_def_205`、`n_op_071/072`
- **修正后引用**：**没有替代值可引用**——文档只给了汇总影响（水平上移 +0.21～+0.27、
  臂排序 Spearman rho=0.5957、10 臂 new-Q1 全部不显著、p 在 0.08–0.35），没有逐臂重打分表。
  所以这一整类只能带限定条件引用：**凡是以"侵蚀显著"为前提的表述，都要写成"自评 judge 下
  显著、独立 judge 下不显著"**。
- **裁决的关键分界**：偏差**不依赖 `u_remind`**（Welch p=0.1309），Phase J 招牌比较在两个
  judge 下同号同结论。所以——
  - **绝对水平**（turn5 y_safety、late_mean_y、全程 y）→ `caveated`
  - **new-Q1 的 t/p/pass** → `caveated`，且是强形式（偏差斜率 +0.072/轮与 new-Q1 的自变量共线）
  - **臂间配对差**（`n_def_206`–`n_def_213` 等）→ `current`
  - **确定性量**（插入次数、token、refusal_rate）→ `current`

### E4 · sycophancy 数据源更换 + 扩样本（触发：2026-09-05）

SYCON-Bench 20 个 item 里 8 个（40%）`correction` 字段有问题，根因是这三个字段由
SYCON-Bench 用 GPT-4o 重新生成、未经证据核验。换成 MMLU（答案来自 MMLU 官方答案表）。

- **superseded（10 条）**：`n_read_075`、`n_read_086`、`n_read_087`、`n_syc_004`–`n_syc_010`
- **修正后引用**：`n_syc_017`（r=0.2894，60 item + turn-1 门槛 + 仅 turn2-5 拟合）是这条线
  **唯一可引用的惯性估计**。
- **惯性 r 的五链**：0.6072（SYCON 独立 judge）→ 0.19（去掉 3 条确认错误）→ 0.72（MMLU 原始
  20 item）→ 0.42（20 item 加门槛）→ **0.29**（60 item 加门槛）。只有最后一个可引用。
- **理由**：0.6072 是 `SYCOPHANCY_KOOPMAN_LOOP_FEASIBILITY.md` 五条前置条件里**唯一"已满足"
  那条**直接引用的数字，审计证明它相当一部分来自 ground truth 有问题的 item 制造的常数序列。

### E5 · Phase J 2 seed → 5 seed（触发：2026-09-03）

- **superseded（42 条）**：`n_def_129`–`n_def_158`、`n_def_165`–`n_def_175`、`n_def_225`
  → 逐条指向 `n_def_176`–`n_def_205`、`n_def_206`–`n_def_213`、`n_def_222`–`n_def_224`、`n_def_226`
- **理由**：文档自己挂了横幅"引用数字请用第十一节"。最优固定臂从 `fixed_t4` 换成 `fixed_t5`，
  固定臂间 late_y 跨度从 0.125 压到 0.075——"哪一轮最好"在 2 seed 下就不稳。
- **一条例外，判 `current`**：`n_def_214/216/218`（2 seed 的 CI 宽度）。CI 宽度是**它自己那次
  实验的属性**，不是对同一个真值的竞争估计；11.4 节正是要把 2 seed 和 5 seed 的宽度并排放，
  才能证明收窄按 √n 生效（实测 0.60 vs 理论 0.63）。

### E6 · Phase B `random_excite` 播种 bug（触发：2026-09-02）—— **计划未列出**

`RandomExciteController` 用 `random.Random(seed)` 建 RNG，而 `run_trajectories_loop` 对每个
`(attack, seed)` 都重新构造控制器，导致 Phase B 全部 300 行**只有两种不同的 5 轮 `u_remind`
序列**。文档字符串写的是 i.i.d. Bernoulli(p)，实际独立激励实现只有 2 个。

- **caveated（2 条）**：`n_def_008/009`
- **判 `current` 的 3 条**：`n_def_107/108/109`——它们**就是**这次 bug 的测量：纯种子混淆
  (+0.0208) 和去混淆后的直接效应 (+0.0208) 一样大，cluster bootstrap CI [−0.025, +0.069] 含 0。
- **为什么不 supersede**：修复只影响未来新采集的数据，已有 Phase B/C/D/E/H 记录不变；
  去混淆估计与新对齐 ARX 拟合出的 `B`(+0.0209) 两种独立方法互相印证，点估计可信。
- **但它是整条线的可辨识性限定条件**：任何基于 Phase B 拟合的论述，都不能说"60 条轨迹的
  随机激励数据"，只能说激励的独立实现只有 2 个。Step 4 的 `intuition_lattice`
  `named_assumption` 应当把这条列进去。

### E7 · `interaction_model_report_valigned.json` 的参照基线是旧对齐值 —— **本次新发现**

这是本次裁决查出的、**此前任何文档都没有记录过**的错误。

- **superseded（2 条）**：`n_op_014`（0.0510）→ `n_base_019`（0.0683）；
  `n_op_015`（0.0430）→ `n_base_018`（0.0684）
- **caveated（1 条）**：`n_op_013`（0.0703，值本身没问题，它承载的比较有问题）
- **机制（已核对三处，不是推测）**：
  1. `analyze_state_action_interaction.py:104-106` 的 `arx_mse`/`richer_mse` 直接从
     `--koopman-fit-report` 指向的 JSON 里读，不重算；
  2. 该产物的 `config` 里 `contemporaneous_v: True` 而
     `koopman_fit_report: outputs/koopman_defense_phaseB_random_excite/koopman_fit_report.json`
     ——指的是**旧**报告，不是 `koopman_fit_report_valigned.json`；
  3. 两份报告实测：旧 arx 0.05099 / richer 0.04300，valigned arx 0.06835 / richer 0.06844。
- **后果**：`koopman_phaseI_policy_closed_form.md` 第 168–169 行的
  "模型本身的 held-out rollout MSE 是 0.0703，还劣于同报告里的参照 ARX（0.0510）和
  `richer_abs_sign`（0.0430）"是一次**混对齐比较**。同对齐下真实对照是 0.0703 vs
  0.0683/0.0684——交互模型仍然更差，但差距是约 **3%**，不是 38%/64%。
- **对论文的影响**：这条比较是"方向学对了不等于预测更准"这个结论的唯一支撑。结论方向保住，
  强度必须从"明显更差"降到"略差、在同一量级内"。**建议同时回改源文档**（`data` 类，不属
  Step 2b 范围，已列在下面第五节升给用户）。

### E8 · LSTM 早停口径（触发：2026-09-03）

旧口径的早停判据**就是被报告的那个 held-out 集合**——用测试指标选停止轮数。

- **superseded（9 条）**：`n_base_009`–`n_base_017`
- **caveated（9 条）**：`n_base_020/021/023/024/026/027/029/030/032`——(a)(b) 两列是泄漏口径，
  **不得作为 LSTM 的性能数字报告**，只能作为把"口径变公平"和"训练数据变少"分开的分解
  （H=8：口径花 +0.0276，少 4 个训练攻击只花 +0.0009）。这一条对 `neurips-write` 的 V8
  （no-leaked-metric，BLOCKING）是直接相关的。
- **修正后引用**：(c) 列 `n_base_022/025/028/031` 与 headline `n_base_033`（LSTM 差 20%–39%）。
- **理由**：`1.88 倍`这个 headline 是 0.0808 vs 0.0430 算出来的，两个数都是旧对齐；
  同对齐 + 乐观口径下是 0.97–1.07 倍，公平口径下才是 20%–39%。方向保住，量级不保。

### E9 · AE baseline 早停追加（触发：2026-09-03）—— **裁决为"不构成作废"**

这一条要写清楚，因为它和 E2 长得像但结论相反。core 任务上早停是**改进**，防御任务上不是：
22 个训练攻击里再切约 4 个做验证集，早停轮数在种子间波动 10 倍以上，`latent_dim=1` 从打平
变成明确更差。文档自己的判读是"验证集太小导致早停失效"，不是"AE 更差"。

- **裁决**：`n_base_038/040/042`（固定 300 epoch）与 `n_base_045/046/047`（早停）**都判
  `current`，互不取代**。headline"AE 与线性打平"用固定 300 epoch 那组；早停那组是
  "core 任务的修法不迁移到这个数据规模"的证据。
- **两组共同背负一条未解决的局限**：重建 loss 在 300 epoch 后仍在 0.09–0.17，没收敛；
  这条局限至今未被解决，只是从"没试过早停"变成"试了但数据规模不支持"。

### E10 · turn-1 基线门槛（sycophancy 线）

未加门槛的估计混进了模型 turn 1 就已答错的 item——这些轨迹没有有效的立场基线。

- **superseded（1 条）**：`n_syc_007`
- **caveated（8 条）**：`n_syc_011`–`n_syc_013`、`n_syc_021`–`n_syc_023`、`n_syc_027/028`
- **修正后引用**：`n_syc_016/017/018`（screening）、`n_syc_024`–`n_syc_026` 与
  `n_syc_029/030`（Phase A 两版提醒设计）。
- **理由**：Phase A 门槛前差值 −0.0292 (t=−1.65)，门槛后缩到 −0.0057 (t=−0.38)——门槛前那点
  非零差值主要由基线本身有缺陷的 item 贡献。
- **配套陷阱（已按文档要求避开）**：门槛必须配合"只在 turn 2–5 拟合斜率"。只加门槛、
  仍连 turn 1 一起回归会制造天花板选择偏差（SYCON 线上曾因此得到一个假的 p<0.05）。

---

## 三、重复记录（Step 4 必须去重，否则 `claim_ledger` 会把一次测量当两条证据）

Step 2a 六条线并行抽取，同一次测量在不同文档里被分别抽了两次。已确认的成对：

| 同一次测量 | id A | id B | 已统一裁决为 |
|---|---|---|---|
| 旧对齐 nu=1,mu=1 的 B | `n_def_010` | `n_def_095` | superseded → `n_def_096` |
| 旧对齐 nu=1,mu=1 的 held-out MSE | `n_def_011` | `n_def_097` | superseded → `n_def_098` |
| Phase C Gramian 条件数 193 / 12.7 | `n_def_014/015` | `n_op_066/067` | caveated |
| Phase A 终轮 0.45 / 0.81 | `n_def_005/006` | `n_op_071/072` | caveated |
| SYCON 独立 judge 惯性 r=0.6072 | `n_read_075` | `n_read_087` | superseded → `n_syc_017` |
| 核心线 joint vs recon 的 46% / 4 倍 | `n_core_009/010` | `n_base_049/050` | caveated |
| `average_word_length_t5` 的 44.6% | `n_core_015` | `n_base_052` | superseded |
| `sentiment_t5` 的 78.1% / 8.1% | `n_core_023` / `n_core_050` | `n_base_053` / `n_base_054` | superseded / current |

---

## 四、Step 2a 遗留的两处，已核实

1. **六条抽取线折进了五个 `line` 标签**。`line` 只有 `core/adversarial_defense/detection/
   readout/sycophancy` 五种，进度表里的 `operator`/`baseline` 两条线被并进了别的桶。
   **不影响分片**——id 前缀完整保留了六条线（`n_def` 227 / `n_read` 92 / `n_op` 73 /
   `n_core` 57 / `n_base` 54 / `n_syc` 30），按前缀切片即可。另有 6 条 `n_base_049`–`054`
   在 `what` 里自标 `MAY NEED RE-TAGGING TO line: core`，确属核心线结果的重复记录（见上表）。
2. **`numbers_unsourced.md` 不存在**。抽样回查未发现追不到出处的条目，判断为确实全部有源；
   若后续发现无源条目，仍按 Step 2a 的规定另记该文件。

---

## 五、升给用户的事项（`data` 类，Step 2b 不自行处理）

1. ~~**E7 需要回改源文档**~~ → **已处理（2026-09-06）**：
   `koopman_phaseI_policy_closed_form.md` 局限第 1 条的对照值已改成同对齐的 0.0683 / 0.0684，
   措辞从"劣于"降为"略劣于……差约 3%"，并附了一段说明这次混对齐是怎么发生的引述块。
   `analyze_state_action_interaction.py` 新增 `_assert_reference_alignment_matches()`：
   拿 fit report 里记的 `config.contemporaneous_v`（缺字段视为旧对齐）与本次运行的
   `--contemporaneous-v` 比对，不一致直接拒绝运行，顺带校验 `nu`/`mu` 一致。
   新增 `tests/test_state_action_interaction_guard.py`（6 个用例，含复现 E7 那次真实配对
   的回归用例），CPU-only，全绿。`n_op_013/014/015` 的裁决不变。
2. **E3 没有逐臂重打分表** → **已拍板并写成执行计划（2026-09-06）**：
   `docs/experiments/independent_judge_reactive_rerun_plan.md`，Sonnet 5 在独立会话执行，
   5 个反应式臂、约 1.5 GPU-小时。固定臂不重跑（日程与 judge 分无关，离线重打分等价于重跑）。
   **在那次重跑归档之前，本文件里 E3 相关的 77 条 `caveated` 裁决不变**；跑完之后需要回到
   Step 2a/2b，把新臂的数字按 schema 补进 `numbers.yaml`，并重新裁决受影响的条目
   （`n_def_021`–`n_def_032`、`n_def_057`–`n_def_059`、`n_def_124`、`n_def_176`–`n_def_205`
   这几组）。执行文档明确禁止 Sonnet 自行改动 evidence 台账。
3. **`caveated` 的引用规则需要你确认**：本文件第 1.1 节把它定义为"可引用但必须同句带限定"。
   如果你希望 Step 4 更保守（`claim_ledger` 只许用 `current`），那么惯性前提（`n_op_065`）、
   Phase J 全部臂级读数、Phase A 执行器权威（`n_def_001`–`006`）都将无法进正文——需要你
   在开始 Step 4 之前明确取舍。
