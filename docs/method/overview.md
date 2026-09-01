# 方法总览

`persona_drift_control/` 实现的流水线分四层，`DATA_COLLECTION_PROTOCOL.md` 定义的是这条
流水线要满足的数学规格，本目录下的 `method/` 记录规格具体是怎么落地成代码的。

```text
被控对象 (plant)          agent 模型 + 模拟用户模型，self-chat
                          -> method/trajectory_generation.md

控制器 (controller)       每轮决定 u_remind（未来扩展 u_gain/u_steer）
                          -> method/controllers.md

测量 (readout)            探针分叉 + 确定性打分函数 -> y_probe
                          -> method/trajectory_generation.md "探针分叉"一节

编排 (orchestration)      跑多少 prompt/seed/条件、写盘 schema
                          -> src/persona_drift/screening.py
```

## 文档索引

- [轨迹生成机制](trajectory_generation.md)：agent 与模拟用户具体如何对话，长度/主题/随机性
  各自怎么控制。
- [控制器抽象](controllers.md)：`u_remind` 取值策略的可插拔接口，已实现的 baseline 控制器，
  以及给 Koopman-MPC 和其他 baseline（split-softmax、激活转向）留的扩展点。
- [Koopman 代理与 ARX baseline](koopman_surrogate.md)：`modeling/` 下的数据加载/状态构造/
  拟合/评测代码，ARX 作为同一套代码的特例，公平对比的具体设计。

## 尚未覆盖 / 已知缺口

- `u_gain`（split-softmax 注意力放大）和 `u_steer`（激活转向）两个通道目前完全未实现，
  `selfchat.py` 里对应字段恒为 0，只是占位。
- MPC 控制器（`control.py::KoopmanMPCController`，用拟合出的 Koopman 代理对 0/1 动作空间做
  短 horizon 穷举求解最优 `u_remind` 序列）**已实现并在对抗防御任务上完整验证**：Phase A→E
  闭环（打赢 zero_control/threshold 两个基线，以更低代价追平 constant_remind），详见
  [../experiments/koopman_defense_pilot.md](../experiments/koopman_defense_pilot.md)。
  人格漂移这条线（`screening.py::_make_controller`）尚未接入它，仍是待办。
- Koopman/ARX 代理已经接到真实采集数据并验证过（对抗防御领域，`nu=1, mu=2`，见下）；
  人格漂移领域的数据仍待正式采集。LSTM baseline 还没实现。详见
  [koopman_surrogate.md](koopman_surrogate.md) 的"已知缺口"。
