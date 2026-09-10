# 文档整理方案（命名基准 + 残留修正 + 结构归位）

**执行者：Sonnet 5。** 这份方案的每一步都有明确规格和验收标准，不需要做判断题。
遇到规格没覆盖的情况**停下来问**，不要自行推断——本方案的大部分风险都来自"顺手多改一处"。

**背景**：项目实际做的任务已经从"人格漂移"转向**抗攻击**和**抗压力**两条线，但文档里
留有旧命名。盘点结果（2026-09-06）：

| 指标 | 数量 |
|---|---|
| md 文档中 `人格漂移`/`人格域`/`persona-drift` 命中 | 75 |
| 其中在四份人格漂移线自己的实验记录里（**白名单，一处都不改**） | 13 |
| `docs/method/` 下指向已放弃线的僵尸待办 | 12 |
| 含 Python 标识符 `persona_drift` 的文件（排除 `outputs/`） | 164 |
| `outputs/` 下的冻结实验产物（gitignore，**一个字都不能改**） | 17 |

---

## 零、总纲：三条硬规则

### 规则 1 · 禁止全局替换

**不许**执行任何形如 `grep -rl 人格漂移 | xargs sed -i 's/人格漂移/抗攻击/g'` 的批量替换。
75 处命中里只有 §二列出的那几行是错的，其余是正确的历史陈述。只按 §二的逐行清单改，
改完逐条 `git diff` 核对。

### 规则 2 · 绝对不改清单

以下内容**在本次整理中一个字都不改**：

1. `persona_drift_control/outputs/**` —— 全部（含 50+ 份 `*_report.md` 和
   `.hydra/hydra.yaml` 快照）。这些是已完成实验的冻结产物，改了就是篡改实验记录，
   而论文流程的 Step 2（证据盘点）正要读它们。已被 `.gitignore` 排除，不会被误提交。
2. `docs/experiments/signal_screening_pilot.md`、`drift_confirmation_pilot.md`、
   `pressure_screening_pilot.md`、`surface_features_backfill.md` —— 这四份**就是**人格漂移
   线的实验记录，全文的"人格漂移"都是准确的。
3. 任何**历史对照句**：模式是"与人格漂移任务的X形成对比"、"复用人格漂移线的Y"、
   "这是为人格漂移线写的草案"、"从人格域移植到Z"。判别标准：**这句话在描述过去发生的事，
   还是在声称项目现在是什么？** 前者保留，后者才改。
4. `persona_drift_control/README.md` 的目录树里
   `selfchat.py  # ① 人格漂移：模拟用户+agent 自聊` —— 这个文件确实属于那条线。
5. `persona_drift_control/resources/PROVENANCE.md` 的
   "used by the persona-drift probe scoring functions" —— `hundred_system_prompts.py`
   确实是人格漂移线的探针打分器，陈述准确。
6. 目录名 `persona_drift_control/` 与 Python 包名 `persona_drift` —— 见 §五，本次不动。

### 规则 3 · 分阶段提交

每个 Phase 结束做一次 `git commit`，commit message 写清改了哪一类。中途出错可以按 Phase 回退。
Phase 之间不要合并提交。

---

## 一、Phase 0 · 建立命名基准表

**产物**：`docs/NAMING.md`（新建）
**风险**：无（纯新增）

后续所有文档、以及论文的 Tier 1 契约（`docs/article/PAPER_EXECUTION_PLAN.md` Step 4 的
keyword lattice）都以这份表为准。先做这一步，否则 Phase 1/2 每改一行都在猜该叫什么。

**表的内容**（照抄，可微调措辞但不要改代号）：

| 代号 | 中文名 | 英文名 | 指什么 | 状态 |
|---|---|---|---|---|
| `core` | 核心 Koopman 建模 | controlled Koopman modeling of LLM output trajectories | 根目录 `src/koopman_ae/`，8 个标量/多变量轨迹任务，`ABLATION_STUDY.md` 八阶段消融 | 活跃 |
| `defense` | **抗攻击** | multi-turn attack resistance / safety-erosion defense | 多轮越狱攻击下的安全侵蚀与 channel-A 提醒注入防御；Phase A→J | 活跃（主线） |
| `stance` | **抗压力** | stance holding under sustained pushback / sycophancy resistance | 用户持续反驳下模型是否放弃正确立场；MMLU "Are You Sure?" 数据源 | 活跃 |
| `benign` | 良性代价对照 | benign helpfulness cost | Phase F 的 MT-Bench 良性会话代价对照 | 活跃（`defense` 的附属） |
| `detect` | 检测支线 | Koopman-based regime detection | 一步预测残差 / 双 regime 对比 / 内容相似度特征 | 收尾 |
| `persona_drift` | 人格漂移 | persona drift | 最初的任务线，screening 三问全挂后放弃 | **已放弃，仅作历史术语** |

**必须写进表里的三条消歧说明**（这是这张表最重要的部分）：

1. **「抗压力」= `stance` 线（sycophancy），不是 `pressure_screening_pilot.md`。**
   `docs/experiments/pressure_screening_pilot.md` 记录的是**人格域**的渐进施压 pilot，
   属于已放弃的 `persona_drift` 线，保留其历史名称。两者都叫"施压"但不是同一件事。
2. **`persona_drift` 作为术语只用于两种场合**：指代那条已放弃的实验线本身，或指代
   目录/包名这个历史遗留标识符。**不得**用它描述项目当前在做什么。
3. **目录名与任务名已经脱钩**：`persona_drift_control/` 这个目录现在装的是 `defense` /
   `stance` / `benign` / `detect` 四条线的代码。名字没改的理由见 §五。

**验收**：`docs/NAMING.md` 存在，六行任务表 + 三条消歧说明齐全。

---

## 二、Phase 1 · 顶层身份表述修正（逐行）

**风险**：低（纯文字）。但必须逐行改，见规则 1。

下面每一条给出文件、行号、现状、改法。行号以 2026-09-06 的版本为准，**改之前先 grep 确认
该行内容匹配**，不匹配就停下来问，不要按行号盲改。

### 2.1 `docs/README.md`

| 行 | 现状 | 改成 |
|---|---|---|
| 1 | `# 人格漂移闭环控制：文档索引` | `# LLM 多轮行为闭环控制：文档索引` |
| 3 | `本目录是 persona-drift Koopman 控制子项目的设计/协议文档集合（代码在 ../persona_drift_control/）。` | 改为说明这是该子项目**四条实验线**的文档集；当前活跃的是**抗攻击**（`defense`）与**抗压力**（`stance`），最初的人格漂移线（`persona_drift`）已放弃；术语基准见 `NAMING.md` |

**同文件内不改的行**：116、138、180、191、198 —— 全部是历史对照句（规则 2.3）。

### 2.2 `persona_drift_control/README.md`

| 行 | 现状 | 改成 |
|---|---|---|
| 1 | `# Persona Drift Control` | `# LLM 多轮行为闭环控制（代码）` |
| 1 下方 | —— | 新增一句：目录名 `persona_drift_control` 与包名 `persona_drift` 是最初任务线的历史遗留，**代码现在服务于抗攻击/抗压力两条线**；未改名的理由见 `../docs/DOC_CLEANUP_PLAN.md` §五 |
| 17（"长出了四条实验线"那段前言） | 只说"长出了四条实验线" | 明确标注①已放弃、②③④的现状，并把②④的名字对齐到 `NAMING.md` 的**抗攻击**/**抗压力** |
| 22（表格行①） | `① 人格漂移 screening（最初的 gate）` | **保留原文**，只在"状态"列末尾追加"（该线已放弃）" |
| 23（表格行②） | `② 对抗防御 Koopman-MPC（主线）` | `② 抗攻击（对抗防御）Koopman-MPC（主线）` |
| 25（表格行④） | `④ sycophancy drift screening` | `④ 抗压力（sycophancy drift）screening` |

**注意同文件 100–102 行还有第二张表**（"哪条线用哪个 sbatch"）：行 100 的"对抗防御"、行 102 的 "sycophancy" 同样按 `NAMING.md` 对齐到**抗攻击**/**抗压力**。

**同文件内不改的行**：39（`selfchat.py` 注释）、56/59（数据集自己的 `persona` 分类名，
是第三方术语）、74/86（`/scratch` 环境路径 `persona_drift_pilot`，见 §五）。

### 2.3 `README.md`（仓库根）

| 行 | 现状 | 改成 |
|---|---|---|
| 27 | 目录树里 `└── persona_drift_control/  # 上述文档描述的实现` | 注释改为说明它现在装的是抗攻击/抗压力两条线的代码 |
| 36 | `该子项目已经跑过四条实验线（人格漂移 screening → 对抗防御 Koopman-MPC Phase A→I → 检测支线 → sycophancy drift screening）` | **保留这句历史顺序**，其后追加一句：当前活跃的是抗攻击与抗压力两条线，术语基准见 `docs/NAMING.md` |

### 2.4 `ABLATION_STUDY.md`

| 行 | 现状 | 改成 |
|---|---|---|
| 5 | `persona_drift_control/ 那条线（人格漂移 → 对抗防御）已经做过多轮消融` | `persona_drift_control/ 那条线（人格漂移 → 抗攻击 → 抗压力）已经做过多轮消融` |

**行 33、424 不改**（`/scratch` 环境路径与目录名引用）。

### 2.5 不需要改的文件

`CODE_DESIGN.md`、`DATASETS.md`、`DATASET_MANIFEST.csv`、
`persona_drift_control/RUNNING_ON_PALMETTO.md`、`persona_drift_control/resources/PROVENANCE.md`
—— 这几份要么不含错误命名，要么含的是准确的历史/路径引用。

**Phase 1 验收**：
```bash
git diff --stat        # 应只涉及 4 个文件
git diff | grep '^[-+]' | wc -l   # 改动行数应在 20 行量级，超过 40 行说明改多了
```

---

## 三、Phase 2 · 过期草案的状态横幅与僵尸待办

**风险**：低。**价值最高的一个 Phase**——它解决的是"读者读到一份文档，不知道它还算不算数"。

### 3.1 统一横幅格式

插在文档标题的**下一行**（若已有类似的 ⚠️ 段落，替换成这个格式，内容保留）：

```markdown
> **状态（2026-09-06）**：<现行 | 部分现行 | 已放弃线的历史草案>。<一句话：哪部分仍在用、
> 哪部分不要照搬>。术语基准见 [`NAMING.md`](../NAMING.md)。
```

`../NAMING.md` 的相对深度按文件位置调整（`docs/` 下直接用 `NAMING.md`）。

### 3.2 需要加横幅的文件

| 文件 | 判定 | 横幅要点 |
|---|---|---|
| `docs/evaluation/EVALUATION_METRICS.md` | **已放弃线的历史草案** | 标题里的"人格漂移闭环控制"是写作时的准确名称，保留；正文当作指标设计方法论与文献出处读，不是现行判据清单；现行判据在 `task/*_FEASIBILITY.md` 与各 `experiments/*_pilot.md` |
| `docs/protocols/DATA_SOURCES.md` | **部分现行** | 标题保留；数据来源清单仍在用，但"人格漂移任务的数据从哪来"这个框定已过时 |
| `docs/protocols/DATA_COLLECTION_PROTOCOL.md` | **部分现行** | 已有区分说明，改成统一格式即可：通道 A–D 定义、readout 约定、screening-先于-建模的 gate 原则**被后续所有线沿用**；具体采集规模属已放弃线 |
| `docs/protocols/KV_INJECTION_MONITORING.md` | **已放弃线的历史草案** | 通道 D（KV 注入）从未实现，且是为人格漂移线设计的 |
| `docs/method/overview.md` | **部分现行** | 四层抽象仍在用；其中人格漂移线的部分已作废 |
| `docs/method/koopman_surrogate.md` | **部分现行** | 同上 |
| `docs/method/controllers.md` | **部分现行** | 同上 |
| `docs/experiments/dose_response_pilot.md` | **已收尾，非活跃** | 已有说明，改成统一格式 |
| `docs/experiments/sycophancy_screening_pilot.md` | **已被取代** | 已有说明（数据源换成 MMLU），改成统一格式 |

### 3.3 僵尸待办（12 处，`docs/method/` 下）

`docs/method/` 里有一批指向**已放弃线**的待办事项，现在读起来像是尚未完成的工作。
逐处在原句**末尾追加**作废标记，**不要删除原句**（保留它记录了当时的设计意图）：

> （该线已放弃，此待办作废）

已定位的位置：

| 文件 | 行 | 原文要点 |
|---|---|---|
| `docs/method/overview.md` | 43 | "没有接进来的是人格漂移线的 `selfchat.py`" |
| `docs/method/overview.md` | 52 | "人格漂移这条线（`screening.py::_make_controller`）尚未接入它，仍是待办" |
| `docs/method/overview.md` | 53 | "人格漂移领域……" |
| `docs/method/koopman_surrogate.md` | 8 | "人格漂移领域仍是合成数据阶段，要等 screening 通过、正式 320 条轨迹采完之后" |
| `docs/method/koopman_surrogate.md` | 58 | "人格漂移领域还没有在真实 `trajectories.jsonl` 上跑过" |
| `docs/method/koopman_surrogate.md` | 62 | "人格漂移领域仍要等 screening 过关、正式数据采出来后" |
| `docs/method/controllers.md` | 37–38 | "把 `KoopmanMPCController` 接入人格漂移这条线" |

`overview.md:17`、`overview.md:23`、`koopman_surrogate.md:15` 是**陈述性**的（描述代码结构或
方法来源），不是待办，**不加标记**。

执行时先跑一遍确认：
```bash
grep -n "人格漂移\|人格域" docs/method/*.md
```
共 12 处，逐处判定"是待办还是陈述"，只给待办加标记。判不准的列出来问，不要猜。

**Phase 2 验收**：`docs/method/` 下不再存在读起来像"未完成工作"的人格漂移条目；
9 份文件的横幅格式一致。

---

## 四、Phase 3 · 文档结构归位

### 3.1 建议：**不搬文件，建双向索引**

盘点结论：根目录的 `README.md` / `CODE_DESIGN.md` / `ABLATION_STUDY.md` / `DATASETS.md` 是
**`core` 任务**（`src/koopman_ae/`）的文档，与代码同级；`docs/` 是 `persona_drift_control`
子项目的文档。**这个划分本身是对的**，问题是它从来没被写下来，导致从 `docs/` 进不去 core 文档、
从根 README 也看不出两套文档的分工。

而搬家的代价不小：`ABLATION_STUDY.md` 在 docs 里被引用约 20 次（多为反引号文字引用），
搬走后读者按名字找文件会找不到。

**所以 Phase 3 做的是加索引，不是搬文件**：

1. **`docs/README.md` 新增一节 `## 仓库级文档（`../`）`**，放在 `## 论文写作` 之前，
   索引根目录四份 core 文档，每份一句话说明它覆盖什么、属于 `core` 线。
2. **根 `README.md` 新增一小段"文档分工"**（放在现有 `docs/` 与 `persona_drift_control/`
   说明那段附近）：根目录文档 = `core` 任务；`docs/` = 子项目四条线；术语基准 = `docs/NAMING.md`；
   论文施工图 = `docs/article/PAPER_EXECUTION_PLAN.md`。

### 3.2 可选微调：只搬一份

`persona_drift_control/RUNNING_ON_PALMETTO.md` 是**仓库通用**的集群操作文档（根 README 要
跨目录链进子项目才能引用它），放在子项目里是错位的。**只有 2 处引用**
（根 `README.md:212`、`persona_drift_control/README.md:75`）。

建议移到 `docs/RUNNING_ON_PALMETTO.md`，同步改这 2 处引用。这一步**独立可选**，
不做也不影响其他 Phase。

### 3.3 不做的事

- 不新建 `docs/core/` 子目录、不搬 `ABLATION_STUDY.md` / `CODE_DESIGN.md` / `DATASETS.md`。
- 不动 `persona_drift_control/README.md` 和 `resources/PROVENANCE.md` 的位置（README 应与
  代码同目录，PROVENANCE 应紧邻它描述的数据）。

**Phase 3 验收**：
```bash
# 所有 md 里的相对链接可解析（18 条）
grep -rno '](\.\{0,2\}/\?[A-Za-z0-9_./-]*\.md)' --include='*.md' docs README.md persona_drift_control/*.md \
  | sed 's/.*](//; s/)$//' | sort -u
# 逐条确认目标文件存在
```

---

## 五、Phase 4 · 代码重命名（**已裁定：不做**）

> **2026-09-10 用户裁定：不改名，也不留 alias shim。** 本节下面的建议（"真要改就同时保留
> `persona_drift` 作为别名 shim"）**不执行**——它与 `.claude/code.md` 的"不写向后兼容 shim"
> 冲突，裁定按 code.md 走，并在那里记了一条只覆盖这一个标识符的**历史包名例外**。
> 本节其余内容保留，作为当时权衡的记录。

目录名 `persona_drift_control/` 与 Python 包名 `persona_drift` 与当前任务已经完全脱钩。
改名是合理的诉求，但**不属于文档整理，且现在做代价过高**。

**规模**：164 个文件含该标识符（`.py` / `.toml` / `.yaml` / `.sbatch` / `.sh`），
波及 import 路径、`pyproject.toml` 的 packages、Hydra 配置的 `_target_`、sbatch 里的路径、
`/scratch/hcao2/envs/persona_drift_pilot` 环境名、checkpoint 路径、`run_config_guard` 的配置比对。

**最硬的反对理由**：`persona_drift_control/outputs/` 下 50+ 份已完成实验的
`.hydra/hydra.yaml` 记录着 `persona_drift.*` 的 `_target_`。改了包名，这些历史产物就不再能被
`analyze_*.py` 重放——**而论文流程的 Step 2（证据盘点）正要读它们**。

**建议**：等论文的 `paper/evidence/numbers.yaml` 落定之后再评估；真要改就同时保留
`persona_drift` 作为别名 shim，让旧 `hydra.yaml` 仍可解析。**在那之前，Phase 1 已经在两个
README 里写清了"目录名是历史遗留"**，读者不会被误导，这已经消除了主要危害。

---

## 六、Phase 5 · 防复发

1. **`docs/README.md` 顶部加一行**：新增文档前先读 [`NAMING.md`](NAMING.md)，任务名以那份为准。
2. **`docs/article/PAPER_EXECUTION_PLAN.md` 的 Step 4** 里，contract 的 keyword lattice
   直接引用 `docs/NAMING.md` 的英文名列——保证论文术语与仓库术语同源。
   （这一条只需在该文档 Step 4 小节加一句交叉引用。）

---

## 七、执行顺序与总验收

```
Phase 0 (NAMING.md)  ──> Phase 1 (逐行改名)  ──> Phase 2 (横幅+僵尸待办)
                                              ──> Phase 3 (索引)  ──> Phase 5 (防复发)
Phase 4 = 不做
```

Phase 0 必须最先。Phase 1/2/3 之间无强依赖，但建议按序做，每个 Phase 一次 commit。

**总验收**（全部 Phase 结束后跑）：

```bash
cd /home/hcao2/autoencoder_koopman_core

# 1. 残留命中数：应从 75 降到 60 上下；每一处都必须落在规则 2 的白名单里
grep -rn "人格漂移\|人格域\|persona-drift" --include='*.md' docs README.md ABLATION_STUDY.md \
  persona_drift_control/README.md | grep -v '^docs/article/'

# 2. outputs/ 零改动
git status --porcelain persona_drift_control/outputs   # 必须为空

# 3. 新文件就位
ls docs/NAMING.md

# 4. 相对链接全部可解析（见 Phase 3 验收脚本）

# 5. 代码零改动
git diff --stat -- '*.py' '*.toml' '*.sbatch'          # 必须为空
```

**第 5 条是最重要的兜底**：本次整理**不应该有任何代码改动**。如果 `git diff` 显示动了
`.py` / `.toml` / `.sbatch`，说明触发了 Phase 4，回退重来。

---

## 八、给执行者的一句话启动语

> 读 `docs/DOC_CLEANUP_PLAN.md`，按 Phase 0 → 1 → 2 → 3 → 5 顺序执行，每个 Phase 一次
> git commit。严格遵守 §零的三条硬规则，特别是"禁止全局替换"和"绝对不改清单"。
> Phase 4 不要做。改动前先 grep 确认行内容匹配，行号对不上就停下来问。
> 判不准某处是"历史陈述"还是"当前身份表述"时，列出来问，不要猜。
