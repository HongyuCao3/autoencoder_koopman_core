# 人格漂移闭环控制：文档索引

本目录是 persona-drift Koopman 控制子项目的设计/协议文档集合（代码在
`../persona_drift_control/`）。信息架构参照
[Pytorch-lightning-Hydra-Optuna-MLflow-Slurm-Project-Template-for-Scientific-Research](https://github.com/HongyuCao3/Pytorch-lightining-Hydra-Optuna-MLflow-Slurm-Project-Tempate-for-Scientific-Research)
的文档分类思路（任务定义/方法/实验/文献），用纯 Markdown 实现，不引入 Quarto 等构建工具。

## 任务与协议

- [DATA_COLLECTION_PROTOCOL.md](DATA_COLLECTION_PROTOCOL.md) — 激励数据采集协议：被控对象、
  readout、输入通道、采集前信号探针（gate）
- [DATA_SOURCES.md](DATA_SOURCES.md) — 数据来源与候选清单：prompt 库、已有轨迹、待采集数据、
  评价用外部数据
- [KV_INJECTION_MONITORING.md](KV_INJECTION_MONITORING.md) — 通道 D（KV 注入）机制与监控协议
- [LLM_LATENT_STATE_FEASIBILITY.md](LLM_LATENT_STATE_FEASIBILITY.md) — 用内部隐状态替代/增广
  Koopman 状态 z_t 的可行性分析（备选方向，暂不默认采用）

## 方法（实现细节）

- [method/overview.md](method/overview.md) — 流水线总览：被控对象/控制器/测量/编排四层的
  代码位置索引
- [method/trajectory_generation.md](method/trajectory_generation.md) — agent 与模拟用户
  具体如何对话：每轮流程、长度控制、主题控制、随机性控制
- [method/controllers.md](method/controllers.md) — 控制器（Controller）可插拔接口，已实现
  的 baseline 控制器，Koopman-MPC 与其他 baseline 的扩展点

## 评价与对比

- [EVALUATION_METRICS.md](EVALUATION_METRICS.md) — 实验完成后用什么指标判断成败
- [BASELINES.md](BASELINES.md) — 待对比的 baseline 清单（控制器层、代理建模层）及对应论文

## 实验

- [experiments/signal_screening_pilot.md](experiments/signal_screening_pilot.md) — 采集前
  信号探针（协议第 7 节的 gate）真实规模作业的状态记录：job ID、怎么查进度、耗时预估、
  作业结束后该做什么。**新开一次对话想知道"之前那个作业现在怎么样了"，看这份文档。**
