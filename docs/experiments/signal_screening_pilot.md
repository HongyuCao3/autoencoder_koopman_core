# 实验记录：采集前信号探针 pilot（真实规模）

记录这次跑在 Palmetto 2 上的作业本身的状态，供换一次新对话（没有本会话上下文）时能接得上——
git 里只有代码和协议，不会有"现在有个作业在跑"这种进行时状态，所以单独记一份。作业跑完/被杀之后
把结论（`overall_pass` 等）也写回这里，不要只留在 `outputs/` 里（那里 gitignore 了，换台机器
或清理 scratch 后就没了）。

## 这是什么实验

`DATA_COLLECTION_PROTOCOL.md` 第 7 节要求的、在正式采集 320 条轨迹之前必须先跑的 gate：
5 个 system prompt × 2 个 seed × 16 轮，`zero_control` 和 `excite_iid` 两种条件各跑一遍
（每个 (prompt, seed) 都跑两种条件，不是只跑一种），共 20 条轨迹，用来回答三个问题：
漂移是否存在／输入是否有效／是否有惯性。任一不过就要先改协议，不能直接开始正式采集。

## 当前作业

| 字段 | 值 |
|---|---|
| Job ID | **15383935**（第三次提交，见下方"历史"） |
| 提交脚本 | `persona_drift_control/environment/run_screening.sbatch` |
| 实际命令 | `python scripts/run_signal_screening.py --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B --device cuda --output-dir outputs/signal_screening`（未传 `--num-prompts/--seeds/--num-turns/--probe-repeats`，用的是协议默认规模：5 prompts, seeds 0 1, 16 turns, 4 probe repeats） |
| 资源 | `work1` 分区，`--gpus a100:1`，`--cpus-per-task 8 --mem 32G` |
| 申请时长 | **`--time 06:00:00`**（根据第二次提交实测的 801s/轨迹算出来的，原来的 3 小时不够，见下方） |
| 提交时间 | 2026-08-29T21:57（提交后立刻开始跑，无排队） |
| 输出目录 | `persona_drift_control/outputs/signal_screening/`（trajectories.jsonl + screening_report.json/md，整个 `outputs/` 被 `persona_drift_control/.gitignore` 排除，不入库） |
| 日志 | `persona_drift_control/environment/slurm_logs/screening-15383935.out`（Slurm 原始 stdout），以及 `persona_drift_control/logs/signal_screening_20260829_220914.log`（`logging_setup.py` 用 loguru 写的，第一行就是完整的运行参数，两边内容重叠但各自独立） |

## 查看状态的方法

```bash
squeue -j 15383935                                  # 是否还在跑、剩余时间
tail -f persona_drift_control/environment/slurm_logs/screening-15383935.out  # 实时进度日志（Slurm 侧）
ls persona_drift_control/logs/                       # 找到这次运行对应的 loguru 文件
wc -l persona_drift_control/outputs/signal_screening/trajectories.jsonl   # 已完成的行数
cat persona_drift_control/outputs/signal_screening/screening_report.md   # 跑完之后才会有
```

`screening.py` 是每条轨迹整体跑完才写一次盘（不是每轮增量写，但会在每条轨迹完成后 `flush()`），
所以行数在下一条轨迹完成前不会变化，属于正常现象；日志里的"starting/finished"两行才是实时的。

- **下次打开这份文档时先做的事**：`squeue -j 15383935` 看作业还在不在。
  - 如果不在了且 `outputs/signal_screening/screening_report.md` 存在：作业正常跑完了，去看
    `overall_pass`，把结论更新到本文件的"结论"一节，再决定要不要开始正式的 320 条采集。
  - 如果不在了但没有 `screening_report.md`：大概率是被 `--time 06:00:00` 到点杀掉了（不太应该
    发生，见下方按 801s/条算出的预算有约 35% 余量，除非实际比第一条轨迹更慢）。看日志里最后一条
    "finished ... in Xs" 重新估算，再调大 `--time` 重新提交。
  - 如果还在跑：`tail` 一下日志文件，看"starting/finished"到第几条了。

## 历史：第一次提交（15382858）为什么被取消了

第一次提交的 job 15382858（2026-08-29T20:48 提交）跑的是**加进度日志之前**的旧版
`screening.py`——生成循环里没有任何 print，只有开头两次模型加载的输出。跑到 50 分钟时
`trajectories.jsonl` 依然是 0 行，日志也从模型加载完之后再没有任何新内容，无法判断是"确实很慢"
还是"卡住了"：

- 想用 `srun --jobid 15382858 --overlap --nodelist node2092 nvidia-smi` 直接看 GPU 利用率，
  但写代码用的这个沙箱的 `srun` 缺共享库（`libhttp_parser.so.2`、PMIx 相关插件），报错退出——
  和 `RUNNING_ON_PALMETTO.md` 记录的沙箱限制一致（只有 `sbatch`/`squeue` 能连上真正的调度器）。
  在真实的登录/OnDemand shell 里 `ssh <节点> nvidia-smi` 应该是可行的，但那需要人在那台机器上
  操作，这次没有走这条路。
- 由于看不出到底是慢还是卡死，且预期"6 分钟/条"、实际超过 50 分钟连第 1 条都没完成，判断继续
  等下去的信息增量太低，选择直接 `scancel 15382858`，用已经加了进度日志的新代码重新提交
  （见上方"当前作业"），用可见性换掉已经花掉的这约 50 分钟 GPU 时间。
- 这次取消**没有排除"其实只是很慢，多等一会儿就会有输出"的可能性**——如果新作业（15383558）
  跑了类似的时长后进度日志显示确实在正常推进（只是每条轨迹比预期慢很多），那说明第一次多半也
  没卡死，只是没有可见度；如果新作业很快就跑出第 1 条轨迹的日志，那第一次大概率是真的卡住了。

## 历史：第二次提交（15383558）证实了"只是慢，没卡死"，但暴露了 `--time` 不够

加了进度日志之后的第二次提交（15383558，2026-08-29T21:39）证实了第一次大概率不是卡死：

- 跑了约 13 分钟后打出 `[1/20] finished character_traits_013__seed0__zero_control in 801s`——
  单条轨迹真实耗时 **801 秒**，`trajectories.jsonl` 也确实写入了这条轨迹的 16 行。说明生成流程
  本身是健康的，只是比协议最初"6 分钟/条"的估计慢了约 2.2 倍。
- 但 801s × 20 条 ≈ 16020s ≈ 4 小时 27 分，超过了当时申请的 `--time 03:00:00`，照这个速度跑下去
  一定会在跑到约第 13～14 条时被 Slurm 强制杀掉，拿不到完整的 `screening_report.md`。
- 权衡之后选择**在它还在正常跑（第 2 条轨迹进行中，约运行 24 分钟）的时候主动取消**，把
  `run_screening.sbatch` 的 `--time` 改成 `06:00:00`（16020s 的基础上留了约 35% 余量）后重新
  提交，而不是等它跑到 3 小时上限被动杀掉再重来——这次取消丢掉的是约 25 分钟（不到 2 条轨迹），
  比放着它跑满 3 小时再重估时间的代价小。

## 排查（2026-08-30，针对三问全挂）

直接读 `trajectories.jsonl`（sandbox 里没有 python，用 perl/awk 手工跑数字）排查为什么三问全不过：

1. **5 个探针 prompt 里有 2 个完全饱和**：`character_traits_013`（"Thank me for each
   question"，打分函数是布尔的 `"thank" in x.lower()`）和 `language_constraints_004`
   （"Always answer with only one sentence"，打分函数是布尔的 `len(sentences)==1`）。
   这两个在全部 64 行（16 轮×2 seed×2 条件）里 `y_probe` 恒为 1.0，`y_probe_sd` 恒为
   0——Qwen3-4B 无论有没有提醒都 100% 满足这两条指令，探针从设计上就测不出任何漂移/控制
   效应。320 行里有 128 行（40%）属于这种情况。
2. **提醒插入机制本身没问题**：抽查了原始行的 `inserted_reminder_text` 字段，`u_remind=1`
   时确实原样插入了 `[Reminder of your instructions: ...]`，`u_remind=0` 时确实是
   `null`。排除了"提醒文本没插进去"这个实现 bug。
3. **去掉那 2 个饱和 prompt 后重新算，结论不变，且不是被稀释出来的假阴性**：
   - Q1 用剩下 3 个连续打分的健康 prompt（teenager_score/正向情感/重复词比例）重算：
     6 条轨迹漂移方向一致，均值降幅 0.0317 vs 阈值 0.0533——比例上跟全量的
     0.019/0.032 几乎一样（都 ≈0.59），说明饱和 prompt 没有系统性偏置 Q1 的判定，只是
     白白浪费了 40% 的算力。
   - Q2 在健康 prompt 上，滞后 1 轮的差值仍是 **-0.0097**（方向反的，n=45/组）；额外测了
     协议没测的"同轮效应"（提醒当轮直接看 y_probe，不等下一轮），差值 **-0.0009**
     （n=48/组）。同轮、滞后 1 轮、滞后 2 轮（Q3 本身）全部测不出效应——这是真实的空
     效应，不是饱和 prompt 稀释出来的假象。

**结论**：两个问题要分开处理——(a) prompt 筛选有缺陷（二值打分函数会饱和，正式 320 条
采集时同样的浪费会更大），(b) 即便排除 (a)，"提醒"这个输入通道在当前规模（16 轮、3 个
可用 prompt、二值 0/1 提醒）下确实测不出因果效应，更像是功效不足而非协议根本性错误——
真实漂移信号本来就小（sd 量级 0.01-0.05），3 个 prompt × 2 seed 很难跳出噪声。

**已做的修复**（`prompt_bank.py` + `analysis.py`，两个改动都在正式采集前落地）：
- `prompt_bank.KNOWN_SATURATED_PROMPT_IDS` 排除了这两个确认饱和的 prompt_id（不改
  `hundred_system_prompts.py`，因为那是 vendored 的，不是我们的）。`load_prompt_bank`
  现在返回 13/28 条（原 14/29），`select_screening_prompts`/正式采集都不会再抽到它们。
- `analyze_screening` 新增 `diagnostics.saturated_prompt_ids`：按 `system_prompt_id`
  分组，本次运行内 `y_probe` 方差恒为 0 的 prompt 会被自动列出来，写进
  `screening_report.md`。这是防御性的——目前只手动验证过这 2 个,全量 40 个 prompt 里
  很可能还有其他没测过的饱和项，靠这个诊断能在未来任何一次运行（screening 或正式
  320 条）里自动暴露，不用再手工翻 jsonl。
- 测试：新增 `test_prompt_bank.py::test_known_saturated_prompts_are_excluded` 和
  `test_analysis.py::test_saturated_prompt_is_flagged_but_varying_prompt_is_not`，
  连同全量测试跑了 Slurm CPU job **15392192**（`run_tests.sbatch`），`35 passed`
  （原 33 个 + 新增 2 个）。

**尚未做、需要决定的**：上面 (b) 的功效不足问题还没解决——要不要重新跑一次规模更大的
screening pilot（更多 prompt/seed 或更多 probe_repeats 压低噪声）来验证漂移/控制效应
是否在更大样本下能跳出噪声，还是先接受当前证据、直接改协议再采集。这一步要再花 GPU
时间，留给下次决定。

## 结论（2026-08-30 补充，作业已完成）

Job 15383935 正常跑完，未被 `--time 06:00:00` 杀掉——20 条轨迹总耗时 10944s（≈3h2m），比
第二次提交按 801s/条估的 4h27m 还快一些（第二批 language_constraints 类别的轨迹普遍比
character_traits 快很多，最快 128s/条）。`screening_report.md`/`.json` 均已产出。

**overall_pass = False，三问全部不过：**

- **Q1 漂移是否存在：不过。** zero_control 下 turn1→last 的 y_probe 均值下降 0.0190，
  小于阈值 0.0320（2×mean y_probe_sd，n=10 条轨迹）——没有观察到有统计意义的人格漂移。
- **Q2 输入是否有效：不过（方向还反了）。** excite_iid 下，`u_remind_t=1` 时下一轮
  y_probe 均值 0.5095，`u_remind_t=0` 时 0.5153，差值 -0.0058，远小于阈值 0.0368
  （n=150 对）——excite 信号没有把 y_probe 往预期方向推，反而略微反向。
- **Q3 是否有惯性：不过。** u_remind_t 对 y_probe_{t+2} 的 OLS 斜率 -0.0055，p=0.9376，
  r=-0.0067（n=140 对）——基本无关系，谈不上惯性。
- 诊断：refusal_rate 1.87%，scorer_failure_rate 0%（生成/打分流程本身健康，问题不在这里）；
  y_probe 均值按类别看 character_traits=0.4805 (sd 0.3868)，language_constraints=0.5596
  (sd 0.4431)。

**下一步：** 按 `DATA_COLLECTION_PROTOCOL.md` 第 7 节，三问任一不过就不能直接开始正式的
320 条采集。三问全挂说明问题大概率不在"运气"（n 都不算小），需要回头检查协议本身，可能的
方向：
- probe 的敏感度/设计是否能测到真实存在的漂移（也可能真的没有漂移，需要另一种诱导方式）；
- `excite_iid` 的输入强度/形式是否足够触发系统性变化（Q2 方向还反了，值得先查这个）；
- 阈值定义（2×mean y_probe_sd）在当前 y_probe 方差这么大（sd≈0.39-0.44）的情况下是否过于
  宽松/严格。
