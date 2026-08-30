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

## 结论（作业完成后补充）

（尚未完成，等 `screening_report.md` 产出后把 `overall_pass` 和三问的具体数值写在这里。如果
`--time 06:00:00` 仍然不够，说明单条轨迹耗时比 801s 的样本还要更长或方差更大，需要按实际日志
重新估算，而不是简单再加时间重提交。）
