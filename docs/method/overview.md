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
- MPC 控制器（用拟合出的 Koopman 代理求解最优 `u_remind` 序列）还不存在，`modeling/koopman.py`
  目前只有拟合/预测/可控性诊断，没有闭环求解器。
- Koopman/ARX 代理还只在合成数据上验证过，没有接到真实采集数据；LSTM baseline 还没实现。
  详见 [koopman_surrogate.md](koopman_surrogate.md) 的"已知缺口"。
