# 计划一：文档 / 代码冗余消除 + 负结果归档（2026-09-10）

**状态**：草案，待 Hongyu 签字后由 Sonnet 执行。**适用**：Sonnet 5（执行）/ Opus 5（C5 核对）。
**规则**：`.claude/global.md` → `.claude/docs.md` → `.claude/code.md`，本文件不重抄。
**前身**：`docs/DOC_CLEANUP_PLAN.md`（2026-09-06，命名整理，Phase 0–3/5 已执行）。本计划不重复它，只做它没覆盖的三件事：**索引瘦身、账本与结果分离、负结果归档**，外加一次代码清单。
**审计依据**：本机克隆 `~/Projects/GitHub/autoencoder_koopman_core`，commit `188aeef`（2026-09-10）。

## 零、审计发现（只列有证据的）

| # | 现象 | 证据 | 违反的既定规则 |
|---|---|---|---|
| F1 | `docs/README.md`（684 行）把每份实验文档的**结论段整段抄进索引**，且状态过期：写"ERGO 计划是当前唯一活计划"、`constraint` 线"未开工" | README L233–330；实际 `constraint` S1 已提交（LEDGER 15772956） | `docs.md` 权威序第 7 条：README 只作入口，不作内容权威 |
| F2 | `docs/LEDGER.md`（580 行）§三·五 里混进了整段分析（S0-0 三闸门表、保真度判定、试点方差拆分） | LEDGER L200–330 | LEDGER 自述"每个作业追加一行" |
| F3 | 37 份实验文档共约 15k 行，**11 份没有状态横幅**；4 份超 700 行（`ergo_multiturn_reliability_pilot` 1598、`adaptive_vs_fixed_claim_plan` 1336、`koopman_defense_pilot` 873、`defense_line_redesign_plan` 716），计划与结果混写 | `grep -m1 状态` 扫描 | `docs.md`：计划 ≤150 行，结果写结果文档；开新计划先给旧的写横幅 |
| F4 | 负结果散在 ≥12 份文档与 LEDGER 行里，论文侧另有 `paper/evidence/superseded.md`（117 条 superseded / 172 条 caveated）；两套口径无互指 | 各文档 | — |
| F5 | `scripts/` 93 个脚本：4 个零引用（`analyze_ergo_entropy_state` `build_len1024_authority_check_sample` `fit_koopman_ae_hydra` `interim_report`）；`analyze_readout_state{,_t1a,_t2,_t3}` 与 `fit_koopman_*` 9 变体是一次性分叉 | 引用计数扫描（docs/sbatch/tests/scripts 四处） | `code.md`：无用代码直接删 |
| F6 | `environment/` 117 个 sbatch 全部平铺，含已收尾三条线的 | `ls environment` | — |
| F7 | 未跟踪目录 `_to_delete/` 含 3 份 08-28 的匿名 md（`GHZCMD8B.md` 等） | `git status` | — |
| F8 | `DOC_CLEANUP_PLAN.md` §五建议改包名时留 alias shim；`code.md` 规定"不写向后兼容 shim" | 两文件 | 规则冲突，需用户裁定 |

## 一、C0 冻结点（Sonnet，10 分钟）

`git tag pre-cleanup-2026-09-10`；`_to_delete/` 三份文件由用户决定删或移出仓库（**不进 git**）。
验收：tag 存在；`git status` 干净。

## 二、C1 索引瘦身（Sonnet，半天）

`docs/README.md` 每条实验文档只留 **≤3 行**：`[链接] — 状态图标 · 一句结论 · 结果在哪一节`。
被删的段落**不丢**：若该段内容在目标文档里已有（多数如此）直接删；若没有，剪贴到目标文档的"结论"节。
同时修正过期状态：`constraint` ▶ 活跃；ERGO ⏸ 挂起；`defense` ⏸ 收尾；`stance` ⏸ 两次空结果；`persona_drift` ⛔。
验收：README ≤ 250 行；每条实验文档恰有一条索引；`grep -c "活计划" docs/README.md` = 1。

## 三、C2 状态横幅补齐（Sonnet，1 小时）

给 11 份缺横幅的文档加 `DOC_CLEANUP_PLAN.md` §3.1 格式横幅（三选一：现行 / 结果档案（冻结）/ 已放弃线的历史草案）：
`koopman_defense_pilot` `koopman_case_study_design` `ergo_multiturn_reliability_pilot` `koopman_detection_design` `koopman_phaseI_policy_closed_form` `ae_baseline_plan` `lstm_baseline_plan` `reminder_history_interaction_plan` `adversarial_screening_thinking_pilot` `surface_features_backfill` `constraint_signal_screening`。
**历史文档不拆分、不改正文**（拆 1598 行的结果档案等于重写历史）。计划≤150 行的规则**只对新文档生效**。
验收：`docs/experiments/*.md` 每份前 12 行含"状态"。

## 四、C3 账本与结果分离（Sonnet，半天）

新建 `docs/experiments/constraint_results.md`（结果档案，按 S0 / S0-0 / 保真度 / S1 试点 / S1 分节）。
把 LEDGER §三·五 中**表格与多段分析**原文剪贴过去；LEDGER 每个作业只留一行 + "结果见 constraint_results.md §x"。
`constraint_signal_screening.md` §十（判据修订记录）保留原地，results 文档只链接它。
验收：LEDGER ≤ 300 行；LEDGER 中不再有 `|---|` 之外的表格；每个 job id 在两处各出现一次。
**以后**：每条线一份 `*_results.md`；LEDGER 行不超过 6 列。

## 五、C4 负结果归档（Sonnet 起草，Opus 核对，1 天）

新建 `docs/NEGATIVE_RESULTS.md`。**每条一行**：`代号 · 一句结论 · 证据指针（文档§ / 产物路径 / LEDGER job id）· numbers.yaml 条目 id（若有）· 是否可被新设定推翻`。
最后一列是本文件存在的理由：计划二若改了信号源位置，哪些负结果**自动失效**、哪些仍成立，一眼可查。
起草清单（Sonnet 逐条核对指针后填，**不新算任何数字**）：

| 线 | 负结果（一句） | 可被新设定推翻？ |
|---|---|---|
| `persona_drift` | screening 三问全挂；10-prompt 功效放大仍空 | 否（线已放弃） |
| `defense` | 独立 judge 天花板 0.91、turn-1 sd=0；基线率 9.8%（Wilson [3.9,22.5]） | 是——换读出位置 / 换被控对象 |
| `defense` | 早期观测对晚期预测力 r≈0（n=255）；攻击文本预测 R²<0 | 是——携带式设计 |
| `defense` | 软 judge RC-1 6/20、RC-3 p>0.7；激活投影 RC-A 非因果不持久；token 概率 G2 饱和 | 部分——RC-A 是在噪声目标上否决的 |
| `defense` | periodic 等代价 ≥ koopman_mpc（new-Q1）；Phase H 有害方向 = v-alignment bug（已修） | 否 / 已修 |
| `defense` | 检测支线方案 4（内容相似度）负；方案 1/3 修 bug 后翻正 | 否 / 已翻 |
| ERGO | k=1 计数预算退化：`fixed_last ≡ always_reset` 116/116 | 否（设定属性） |
| ERGO | append 打破恒等式但抹掉权威；E2/E3 的机制主张错 | 否 |
| ERGO | 熵读出控住斜坡后衰减 97%；`b₂` CI 跨 0（MDE 4.5×，判不可判定） | 部分——`b₂` 是"不可判定"不是"无" |
| `stance` | Phase A 执行器权威两次空结果（SYCON、MMLU） | 是——但线已挂起 |
| `constraint` | 旧 harness 下 K1/K3 不过——**harness 造成**，保真 harness 全过 | 已推翻 |
| `constraint` | G-FID 只过一半（t1=0.75 > 0.60）；`tuple_294_1` 三次咬 cap | 记录 |
| `core` | AE 第一阶段消融跑在 `epochs=200`，训练预算 artifact（`caveated`） | 是——需早停复核 |

与 `paper/evidence/numbers.yaml` 的关系：**只互指，不复制数字**（`.claude/paper.md`：不碰 `paper/`）。
验收：每行五列齐全；Opus 抽查 5 条指针能打开；`grep -c "^|" NEGATIVE_RESULTS.md` ≥ 15。
**以后**：任何闸门判 FAIL 的同一次 commit 里追加一行。

## 六、C5 归档搬家（Sonnet，1 小时，机械判据）

对 `docs/experiments/*.md` 逐份判：**含带数字的结果节 → 原地冻结（只加横幅）；纯计划且结果在别处 → 移入 `backup/` 并在其 README 表加行。**
预判可移：`reminder_history_interaction_plan`（107 行）、`ergo_fidelity_restoration_plan` 若其结果已迁入 `ergo_multiturn_reliability_pilot`。其余大概率冻结。判不准的列出来问。
验收：`backup/README.md` 表每行四列齐全；被移文档的所有入链仍可解析（跑 `DOC_CLEANUP_PLAN.md` §3 的链接脚本）。

## 七、C6 代码清单（Sonnet，半天；**不删复现路径**）

1. `scripts/README.md`（新）：93 个脚本按线 × 状态（活跃 / 冻结-复现用 / 候删）三列列表，一行一个。
2. 四个零引用脚本：**先在集群** `grep -rl <名> /home/hcao2/autoencoder_koopman_core/persona_drift_control/outputs/*/run_*.json`，零命中才删；有命中则标"冻结"。本机 `outputs/` 为空，**不得据本机判断删除**。
3. `environment/`：新建 `archive/` 子目录，移入 `defense` / `stance` / `persona_drift` 三线的 sbatch（按文件名前缀与 LEDGER 对照）；ERGO 与 `constraint` 的留原地。sbatch 不被 import，移动零风险；LEDGER 里的作业名不变。
4. **不动**：`control.py` `controller_cli.py` `modeling/evaluate.py` `dataset.py`、所有守卫与元测试、`tests/`。
5. F8 冲突交用户：包名 `persona_drift` 是否改？建议**不改**（`DOC_CLEANUP_PLAN.md` §五的理由仍成立），并在 `code.md` 加一句"历史包名例外"。
验收：`git diff --stat -- '*.py'` 只含删除的零引用脚本；`pytest` 全绿；`scripts/README.md` 行数 = 脚本数。

## 八、顺序、成本、与计划二的接口

```
C0 → C1 → C2 → C3 → C4 → C5 → C6   （每步一次 commit，共 7 次；总计约 3 人天，0 GPU）
```
C1–C3 与 S1 作业同时进行互不干扰；**C4 必须在计划二开 S3 之前完成**——S3 的臂表要引用"哪些负结果在新设定下仍成立"。
C6 可拖到截稿后。

## 九、不做的事

不拆历史长文档；不改任何 `outputs/`；不重算任何数字；不改包名；不动 `paper/`；不为整理而新开实验。
