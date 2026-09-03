# 方法总览

`persona_drift_control/` 实现的流水线分四层，`DATA_COLLECTION_PROTOCOL.md` 定义的是这条
流水线要满足的数学规格，本目录下的 `method/` 记录规格具体是怎么落地成代码的。

```text
被控对象 (plant)          agent 模型 + 模拟用户模型，self-chat
                          -> method/trajectory_generation.md

控制器 (controller)       每轮决定 u_remind（u_steer 已在防御线单独实现，u_gain 未实现）
                          -> method/controllers.md

测量 (readout)            探针分叉 + 确定性打分函数 -> y_probe
                          -> method/trajectory_generation.md "探针分叉"一节

编排 (orchestration)      跑多少 prompt/seed/条件、写盘 schema
                          -> src/persona_drift/screening.py（人格漂移线）
                             adversarial_screening.py / benign_screening.py /
                             sycophancy_screening.py（后来长出的三条线，
                             共享 screening_common.py + trajectory_runner.py）
```

这四层的抽象是为人格漂移线写的，但对抗防御、良性 helpfulness、sycophancy 三条线沿用了同一套
分层（换掉 plant 的输入序列、readout 的判分器和 controller 的目标，骨架不变）——各线的状态见
[`../README.md`](../README.md)。

## 文档索引

- [轨迹生成机制](trajectory_generation.md)：agent 与模拟用户具体如何对话，长度/主题/随机性
  各自怎么控制。
- [控制器抽象](controllers.md)：`u_remind` 取值策略的可插拔接口，已实现的 baseline 控制器，
  以及给 Koopman-MPC 和其他 baseline（split-softmax、激活转向）留的扩展点。
- [Koopman 代理与 ARX baseline](koopman_surrogate.md)：`modeling/` 下的数据加载/状态构造/
  拟合/评测代码，ARX 作为同一套代码的特例，公平对比的具体设计。

## 尚未覆盖 / 已知缺口

- `u_gain`（split-softmax 注意力放大）仍完全未实现；`selfchat.py` 里的 `u_gain`/`u_steer`
  字段恒为 0，只是占位。但 `u_steer`（激活转向）本身**已经在对抗防御线上实现并跑过**——
  `chat_model.py` 的逐层残差流加 `α·v`、`activation_direction.py` 的 diff-in-means 方向标定、
  `dose_response.py` 的 α 扫描，结果见
  [`../experiments/dose_response_pilot.md`](../experiments/dose_response_pilot.md)。
  没有接进来的是**人格漂移线的 `selfchat.py`**，不是这个通道本身。
- MPC 控制器（`control.py::KoopmanMPCController`，用拟合出的 Koopman 代理对 0/1 动作空间做
  短 horizon 穷举求解最优 `u_remind` 序列）**已实现并在对抗防御任务上跑完 Phase A→I**：
  Phase A→E 打赢 zero_control/threshold 两个基线、以更低代价追平 constant_remind，但
  Phase G 补上 `periodic` 这个零建模的固定日程基线后**没有打赢它**，Phase H/I 修掉 v 对齐
  bug 后仍然没有（结论是设定层面的：那个评测里不存在分配问题）。当前在跑的 Phase J 换成
  预算约束设定重做这个对比，见
  [../experiments/budget_constrained_defense_plan.md](../experiments/budget_constrained_defense_plan.md)
  与 [../experiments/koopman_defense_pilot.md](../experiments/koopman_defense_pilot.md)。
  人格漂移这条线（`screening.py::_make_controller`）尚未接入它，仍是待办。
- Koopman/ARX 代理已经接到真实采集数据并验证过（对抗防御领域，`nu=1, mu=2`）；人格漂移领域
  的数据仍待正式采集。LSTM 与 AE（encoder-decoder）两个 baseline **都已实现并跑完**（分别是
  负结果和打平），详见 [koopman_surrogate.md](koopman_surrogate.md) 的"已知缺口"——那一节是
  这三条的权威版本，本节只做索引，两边不一致时以它为准。
