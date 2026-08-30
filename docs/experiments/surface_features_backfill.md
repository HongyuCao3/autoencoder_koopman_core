# 实验记录：signal_screening_pilot 数据的免费表层特征回填（CPU-only）

记录这次分析的目的、代码位置和结论，跟 `signal_screening_pilot.md`/`drift_confirmation_pilot.md`
是同一类"供跨会话接续"的记录，但这次不涉及 GPU 作业排队/续跑，跑完就有结果，不需要"下次打开先做
什么"那一节。

## 这是什么，为什么做

`ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md`/`OPEN_DATASETS_AND_TRAJECTORY_ACCELERATION.md` 都
提到一个未验证的风险：把 `y_probe` 换成免费表层特征（词数、POS 频率、可读性...）当 readout，
不能假设它就一定测得到漂移——y_probe 在 `signal_screening_pilot.md` 里已经测不出显著漂移
（Q1 不过），换一个不同构造的指标完全可能面临同样的"空效应"。这次分析用零 GPU 成本核实这个
风险：不生成任何新文本，只对 `signal_screening_pilot` **已经跑完**的20条轨迹（10条
zero_control）里的 `agent_message` 文本，回填几个免费特征，重跑一遍跟 `analysis.py`
`q1_drift_trend` 完全一样的检验方法（按轨迹算全序列 OLS 斜率、按 prompt 聚合、单样本 t 检验），
看这些指标是否也测不出漂移。

## 代码

- `src/persona_drift/surface_features.py::extract_surface_features`：纯函数，输入一段文本，
  输出 `num_tokens`/`num_sents`/`avg_word_len`/`ttr`（type-token ratio）/
  `adj_ratio`/`noun_ratio`/`verb_ratio`/`adv_ratio`（NLTK Penn Treebank POS 频次）/
  `vader_sentiment`（NLTK VADER 词典打分）九个特征。只用 `environment/setup_env.sh` 已经装好
  的 nltk（POS tagger + VADER lexicon），不需要 spaCy/rstfinder，是
  `ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md` 第4节"~50维特征"里可以零新增依赖直接做的一个
  子集，不是完整复现。
- `tests/test_surface_features.py`：5个 CPU 单测（无需GPU/nltk外网下载，走的是env里已缓存的
  语料），随 `pytest -q` 一起跑，job **15394039** 验证过：`56 passed`（含此前已有的51个）。
- `scripts/backfill_surface_features.py`：CLI，读 `trajectories.jsonl`（只读，不改动原文件），
  对每行 `agent_message` 算九个特征，写一份新文件
  `trajectories_with_surface_features.jsonl`，再对 zero_control 子集按特征逐个跑漂移检验，
  产出 `surface_features_drift_report.json`/`.md`（这三个输出文件都在 `outputs/` 下，
  跟其他产出一样被 `.gitignore` 排除，不入库，结论摘要写在下面）。
- `environment/run_backfill_surface_features.sbatch`：CPU-only（无 `--gpus`），2核4G、
  15分钟上限，跟 `run_screening.sbatch`/`run_drift_confirmation_pilot.sbatch` 完全独立提交，
  不会跟它们抢 GPU 分配——实测提交时 `drift_confirmation_pilot`（job 15393201）正在跑，
  这次 job（**15394052**）被调度到另一个节点（node0598 vs node0406），两边互不影响，
  15393201 全程没有中断，进度正常推进。

## 结果（job 15394052，2026-08-30，作业已完成）

对 `outputs/signal_screening/trajectories.jsonl`（20条轨迹，10条 zero_control，5个prompt×
2seed）：

| 特征 | turn1→last 均值变化 | n_prompts | 负/正斜率prompt数 | slope-vs-0 的 p 值 | 显著（p<0.05）？ |
|---|---:|---:|---|---:|---|
| **avg_word_len** | **+0.1463** | 5 | **5/0（全部一致）** | **0.0412** | **是** |
| num_tokens | -61.70（变长） | 5 | 0/5（全部一致） | 0.2455 | 否 |
| ttr | +0.0936 | 5 | 3/2 | 0.2558 | 否 |
| adj_ratio | -0.0111 | 5 | 4/1 | 0.6599 | 否 |
| num_sents | -0.5000 | 5 | 3/2 | 0.5572 | 否 |
| noun_ratio | -0.0003 | 5 | 3/2 | 0.4959 | 否 |
| verb_ratio | -0.0099 | 5 | 2/3 | 0.1868 | 否 |
| adv_ratio | -0.0130 | 5 | 1/4 | 0.2003 | 否 |
| vader_sentiment | -0.0607 | 5 | 2/3 | 0.3880 | 否 |

**核心发现：`avg_word_len`（平均词长）在 zero_control 下有统计显著的下降趋势
（p=0.0412），且5个prompt的斜率方向完全一致（全部为负）**——这是在 `y_probe` 判定"没有漂移"
（`signal_screening_pilot.md` Q1 不过）的**同一批文本**上测出来的。`num_tokens`（回复变长）
虽未达到 p<0.05（n=5 统计功效本来就低），但5个prompt方向也完全一致，是同一方向上的次强信号。
其余7个特征都测不出方向一致或显著的趋势。

**解读，需要谨慎，不要过度声称**：这**不等于**确认了协议意义上的"人格漂移"真实存在——更合理
的解读是随对话轮次推进，模型回复变长、平均用词变简单（词长下降），这更像是一种**通用的"对话
变随意"效应**，和 system prompt 规定的具体人格/格式约束（`y_probe` 测的东西）是两个不同的构造。
也就是说：`y_probe` 测不到东西，不代表这批文本里什么都没变，但变化的东西未必是"人格保持"本身。
这个区分本身就是本次分析最重要的产出——此前"换个免费指标不一定测得到"的顾虑，现在有了一个
具体、可复现的反例（`avg_word_len` 测到了，`y_probe` 没测到），值得在下一步设计里正视，而不是
简单假设"换指标就能解决 signal_screening_pilot 的空效应问题"。

## 下一步（留待决定，本次未做）

- 要不要在正式的 `analysis.py`/`selfchat.py` 里把 `avg_word_len`（或全部9个特征）正式接入
  为常规产出的一列，而不只是这次的离线回填脚本——如果接入，`y_formality`/`y_sentiment` 现在
  一直是 `None` 的两个占位列，`vader_sentiment` 可以先填其中一个（性质相近但不是同一个打分器，
  需要在文档里说明是替代品不是原计划的 Cardiff RoBERTa）。
- 要不要在 `drift_confirmation_pilot`（更大样本、10个prompt）跑完之后，对那批数据也做一次
  同样的回填，看 `avg_word_len` 的显著性在更大样本下是增强还是消失——现在 n=5 个 prompt 的
  显著性本身统计功效有限，值得在更大样本上复核，而不是直接采信这一次的结果。
