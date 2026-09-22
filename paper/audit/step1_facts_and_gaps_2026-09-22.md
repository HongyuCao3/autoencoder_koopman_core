# Step 1 事实核查与新发现的缺口（2026-09-22）

> **这是什么**：`PLAN_finish_2026-09-22.md` **Step 1** 的执行记录——四个待查事实（F1–F4）的答案、
> 出处，以及查证过程中发现的、计划里没有的缺口。
> **状态**：证据档案，**未改动任何 `sections/*.tex`、`tables/*.tex`、`evidence/*.yaml`**。
> 口径与优先级见 `../../.claude/global.md`；术语见 `../../docs/NAMING.md`。

---

## 一、F1 — 向量属性怎么建模

**答：一个联合模型，向量读数整体堆进延迟窗口；不是每坐标一个标量模型。**

| 环节 | 事实 | 出处 |
|---|---|---|
| 读数 | `y_t = [word_count_norm, avg_word_length_norm(, comma_count_norm)] ∈ R^m` | `src/koopman_ae/core.py:212`；`configs/dataset/vector_count_stage{1,2}_t10.yaml` |
| 状态 | `z_t = [y_t, y_{t-1}, …, u_t, u_{t-1}, …]`，**每个 lag 位放整个 m 维向量** | `core.py:169-178` `_state_at_turn` |
| 目标与控制 | `r` 亦为 m 维；`u_t = r − y_t` 逐维 | `core.py:149-167` `_control_from_y_r` |
| 读出 | `predict_y` 取 `z` 的前 `output_dim` 维 → **C = [I_m, 0, …, 0]，每坐标一行** | `core.py:355-358`、`core.py:1028-1030` |
| Table 1 的 Joint 列走的就是这条路径 | `eval_surrogate_rows.py` 调 `build_augmented_state_dataset` | `scripts/eval_surrogate_rows.py:37-45` |

**对 Step 2a 的影响（比计划写的大）**：`sections/03_method.tex:47-56` 有**四处维度**在向量情形下不成立——
$s_t\in\mathbb{R}^{L+1}$、$C=[1,0,\ldots,0]$、$K\in\mathbb{R}^{(L+1)\times(L+1)}$、$B\in\mathbb{R}^{(L+1)\times 1}$。
要么逐处加"标量情形"限定，要么改写。这同时把 `PLAN_finish` 的 open decision 3（字母 $C$ 一词三义）
从"可选"变成"几乎必须"：2a 之后 $C$ 是矩阵，与 $\mathcal{C}$（命令域）、$\mathcal{C}_t$（候选集）撞得更难读。

---

## 二、F2 — 每个属性由谁打分

**答：七列不是同一个打分方；`sections/04a_setup.tex:13` 现在这句是事实错误，2b 要"改句"不是"加句"。**

| Table 1 列 | 读数来源 | 类别 | 出处 |
|---|---|---|---|
| Sentence len. | 词数程序 | 程序 | `DATASETS.md`；`configs/dataset/sentence_length_t10.yaml` |
| Joint-2 / Joint-3 | 词数 / 平均词长 / 逗号数程序 | 程序 | `configs/dataset/vector_count_stage{1,2}_t10.yaml` |
| Sentiment | Cardiff 归一化读出 | 外部分类器 | `configs/dataset/sentiment_t5.yaml:4` |
| Defense | **被控模型自判**（judge = agent = Qwen3-4B） | 打分模型（自判） | `evidence/numbers.yaml` n_main_136–151 `judge_kind=self` |
| Constraint | 报告用**独立 Qwen3-14B**；in-loop 驱动控制器的是 Qwen3-4B-Instruct-2507 | 打分模型（独立） | `docs/experiments/behavioral_surrogate_rows_results.md:77`；n_loop_048 |
| CEFR | **三个 ModernBERT 分类器**（官方选择规则）+ MeaningBERT 保义 | 外部分类器 | `persona_drift_control/src/persona_drift/tsar_cefr_readout.py:1-56` |

**两处具体错误**：

1. **"which both produces and scores the self-built trajectories"** — 只有 `defense` 一列是被控模型自己打的分；
   其余六列分别由程序、Cardiff、Qwen3-14B、ModernBERT 打。
2. **"runs on a single model, Qwen3-4B-Instruct-2507"** — 生成方也不统一：
   `tsar_cefr` 的 agent 是 `Qwen/Qwen3-4B`（`persona_drift_control/environment/run_tsar_cefr_*.sbatch`），
   `constraint` 才是 `Qwen3-4B-Instruct-2507`（`run_sequor_s0_0_arm.sbatch:70`）。

**第三件正文没说的事**：CEFR 列建模的是 `level_expected`（Σ k·p_k 的温度软化**期望级**，控制读出），
不是 shared task 官方的 argmax 级别（`tsar_cefr_readout.py:5-31`）。读者默认会当成官方指标。
2b 那句话正好是安放它的位置。

---

## 三、F3 — 约束满足的逐子任务定义

**答：仓库里没有。** `evidence/numbers_benchmark.yaml` 只有百分数，`sections/04a_setup.tex:36-37`
的 `% TODO(data)` 是唯一出口。**只能等对方给。** 这条同时卡住 D30 的可行集桥接。

---

## 四、F4 — 等代价对照的四列

**答：四列 = `defense` / `constraint` / `gsm8k_sharded` / `tsar_cefr`**
（`docs/article/MAIN_TABLE_DESIGN.md:206-210`）。**它们不是"额外的四个自建任务"**——
其中三个（defense / constraint / CEFR）就是 Table 1 七列里的三列，第四个 `gsm8k_sharded` 是附录 A2 那列。

| 列 | 对"最优等代价固定日程"的主量 | 判读 | 出处 |
|---|---|---|---|
| `defense` | late_y **−0.0250** [−0.0500, −0.0042]；terminal **−0.0688** [−0.1313, −0.0063] | **输**（按攻击单位 CI 排除 0；按轨迹单位含 0，**两单位规定必须同时引**） | n_loop_040 / n_loop_039 |
| `constraint` | **+0.0014 ± 0.0145 (n=3)**，0.05×MDE，CI 含 0；同批行里提醒本身买到 +0.1163 ★ | **干净打平** | n_loop_048 / n_loop_050 / n_loop_051 |
| `tsar_cefr` | 同家族内 vs `fixed_ladder` −0.0073（含 0）、vs `greedy_reactive` +0.0096（含 0，符号不利）；等代价 **3.37 步 / 322 token 对 1.03 步 / 100 token** | **打平但代价上被开环阶梯支配** | `MAIN_TABLE_DESIGN.md:268`；`docs/experiments/tsar_cefr_results.md` §Table 2 |
| `gsm8k_sharded` | **构造上恒等于 `fixed_last`**（日程可分性 = 1），裁决规定不编数 | 未测 | `MAIN_TABLE_DESIGN.md:133` |

**`sections/04d_analysis.tex:15-16` 的三个毛病**：

1. **"four additional self-built tasks"** — 不是额外任务，与 `04a_setup.tex:8`"七个自建任务"打架。
2. **"ties"** — 四列里只有 constraint 是打平；defense 在主口径下**显著输**，gsm8k 是构造恒等（未测），
   CEFR 打平但代价被支配。`../../.claude/paper.md` 的诚实性红线要求"打不赢等代价日程"进正文。
3. **"depends little on when it is issued"** — 只对 defense / gsm8k 成立。CEFR 的机制是
   **贪心已收满 98%、规划余量 0.0130 对 MDE 0.095**（`docs/NAMING.md` tsar_cefr 行），不是"何时下指令"。

---

## 五、计划之外发现的缺口

### G1（P0）Table 1 `defense` 列的同址义务一件未落

该列同时是 `judge_kind=self` **和** `n_seed=2`（n_main_136–151 的 `supersede_reason`）。
按 `../../.claude/global.md` 的具名例外与 `docs/article/MAIN_TABLE_DESIGN.md` 裁决 6，
该列每次出现须同址携带：局限句 + `judge_kind=self` 标注 + `n_seed=2` + 重采样单位 +
"本列判读为本设计分辨不出来"。

**现状**：`tables/tab_prediction.tex` 的 caption 与 §4.2 正文**一样都没有**；
`tools/build_tables.py:53-56` 自己记着"用户选择不加 in-text caveat"。
另：该列给 Last-turn 的 0.067 加粗，而六行全距仅 0.009
（`docs/experiments/behavioral_surrogate_rows_results.md:66-68` 判"读出没量程"）——加粗在读者眼里就是排名。
**`PLAN_finish` 七步里没有任何一步处理它。需要用户裁决。**

### G2（P0）Step 6a 与"main.bib 由用户手工维护"冲突

Step 6a 要求新增 5 个键、修 6 个 `unknown*` 键、合并两条 GenCtrl。核对结果：**缺的是 6 个键，不是 5 个**——
`main.bib` 现有 20 个键，确无 `korda2018koopman` / `kong2024recontrol` / `wang2026tmpc` /
`akrout2026distinguishability` / `dmd2026safety`，**`li2024instability` 也不在**（Step 6a 漏列，它是 Intro P2 的引用之一）；`unknown2024activationtraits` 等 6 个 `unknown*`
与 `cheng2026genctrl` / `unknown2026genctrl_dup` 都在。
但 2026-09-17 用户裁决 bib 由本人手工填。**Step 6a 整条应交回用户**；Step 5a（Related Work）按
`../../.claude/paper.md` 的分层本就是用户的活，会一起停。

### G3（P1）Step 7 的终检命令在本机跑不了

`pdftotext` 不在环境里（`~/.local/bin` 只有 TeX Live 一套 + `latexmk`/`pdflatex`）。
需要装，或换 `pdfgrep` / `mutool draw -F txt`。

### G4（P1）"Table 2" 一词两义

论文里 Table 1 = 预测表、Table 2 = 控制表；项目文档里 "Table 2" 指**闭环等代价表**（A4 要填的那张）。
`PLAN_finish` Step 6b 的"Table-rounded 5.3"指论文 Table 2，Step 4a 的 A4 内容对应文档 Table 2。
**填 A4 的人要先知道这件事。**

### G5（P1）A7 内部两句并排读像互斥

`sections/appendix/A7_command_channel.tex:5` 说命令等于跟踪误差；:7 说命令在每个 transition 上等于目标
（2268/2268）。两句各有出处（`docs/NAMING.md` core 行 / 采集日志的标量命令列），但并排放着像自相矛盾。
Step 7 的一致性复查应覆盖。

---

## 六、已核对无误

- 控制线四个数字与 `tables/tab_control.tex` 逐位对得上：41.4−26.9 = **14.5**、47.2−32.4 = **14.8**、
  41.4−36.1 = **5.3**（未舍入 5.25）、47.2−41.5 = **5.7**；
  `sections/01_introduction.tex:56` 与 `sections/04c_control.tex:12,16` 口径一致。Step 5c 照抄即可。
- 命令通道的诚实性已经到位：`04d_analysis.tex:18-19` + `A7_command_channel.tex` 明写
  "四列上无法估计独立动作效应"，与 `docs/NAMING.md` core 行的警告一致。
- Step 3 的图 1 草稿已有人在做（未入库：`figs/method_fig/sketches/`）。

**未核对**：页预算（body 至 p.9 line ~449 / 上限 486）——需编译才能验，本轮按"不改动"未跑。
**未取得**：`PLAN_finish` 引用的 `claude/fulltext_consistency_audit_2026-09-22.md` 不在仓库
（只有编译产物 `build/main_consistency_2026-09-22.pdf`），故"审计标 done 的都已在树上"这句未独立复核。
