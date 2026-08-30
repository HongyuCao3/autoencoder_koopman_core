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
| Job ID | **15383558**（第二次提交，见下方"历史"） |
| 提交脚本 | `persona_drift_control/environment/run_screening.sbatch` |
| 实际命令 | `python scripts/run_signal_screening.py --agent-model Qwen/Qwen3-4B --user-model Qwen/Qwen3-4B --device cuda --output-dir outputs/signal_screening`（未传 `--num-prompts/--seeds/--num-turns/--probe-repeats`，用的是协议默认规模：5 prompts, seeds 0 1, 16 turns, 4 probe repeats） |
| 资源 | `work1` 分区，`--gpus a100:1`，`--cpus-per-task 8 --mem 32G` |
| 申请时长 | `--time 03:00:00` |
| 提交时间 | 2026-08-29T21:39（提交后立刻开始跑，无排队） |
| 输出目录 | `persona_drift_control/outputs/signal_screening/`（trajectories.jsonl + screening_report.json/md，整个 `outputs/` 被 `persona_drift_control/.gitignore` 排除，不入库） |
| 日志 | `persona_drift_control/environment/slurm_logs/screening-15383558.out` |
| 与上一次的区别 | 这次跑的是**加了逐条轨迹进度日志**（开始/结束时间戳 + `handle.flush()`）之后的 `screening.py`，日志里应该能实时看到跑到第几条、每条耗时多久，不会再像上一次那样完全没有可见度 |

## 查看状态的方法

```bash
squeue -j 15383558                                  # 是否还在跑、剩余时间
tail -f persona_drift_control/environment/slurm_logs/screening-15383558.out  # 实时进度日志
wc -l persona_drift_control/outputs/signal_screening/trajectories.jsonl   # 已完成的行数
cat persona_drift_control/outputs/signal_screening/screening_report.md   # 跑完之后才会有
```

`screening.py` 是每条轨迹整体跑完才写一次盘（不是每轮增量写，但现在会在每条轨迹完成后
`flush()`），所以行数在下一条轨迹完成前不会变化，属于正常现象；进度日志里的"starting/finished"
两行才是实时的。

- **下次打开这份文档时先做的事**：`squeue -j 15383558` 看作业还在不在。
  - 如果不在了且 `outputs/signal_screening/screening_report.md` 存在：作业正常跑完了，去看
    `overall_pass`，把结论更新到本文件的"结论"一节，再决定要不要开始正式的 320 条采集。
  - 如果不在了但没有 `screening_report.md`：大概率是被 `--time 03:00:00` 到点杀掉了。看日志里
    最后一条 "finished ... in Xs" 算出单条轨迹的真实耗时，改大 `run_screening.sbatch` 的
    `--time` 后重新 `sbatch` 提交。
  - 如果还在跑：`tail` 一下日志文件，看"starting/finished"到第几条了，正常来说不应该再像
    之前那样完全没有输出。

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
  两种情况都要在这里补一句结论。

## 结论（作业完成后补充）

（尚未完成，等 `screening_report.md` 产出后把 `overall_pass` 和三问的具体数值写在这里，同时
补上"15383558 是否证实了 15382858 只是慢/还是真卡死"的判断。如果这次也被 `--time 03:00:00`
杀掉且没跑完，说明这个规模/配置在当前推理速度下不可行，需要用日志里实测的单条耗时重新评估
`--time`，而不是简单地再重提交一次。）
