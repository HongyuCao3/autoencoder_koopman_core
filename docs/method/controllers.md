# 控制器（Controller）抽象

`src/persona_drift/control.py` 把"这一轮 u_remind 取什么值"从生成/测量逻辑
（`selfchat.py::run_trajectory`）里抽出来，做成一个可插拔的接口。这是 2026-08-29 为了让
Koopman-MPC 未来能和其他 baseline 控制器公平对比而做的重构（背景见
`../BASELINES.md`）——重构前 `u_remind` 的取值是写死在 `run_trajectory` 内部的 if/else。

## 接口

```python
class Controller(Protocol):
    name: str
    def next_u_remind(self, turn: int, history: list[dict[str, Any]]) -> int: ...
```

`run_trajectory` 每轮调用一次 `controller.next_u_remind(turn, rows)`，`rows` 是这条轨迹里
已经写完的历史行（按轮次顺序，最新的在最后）。`name` 字段写入输出行的 `excitation_design`
列，供 `analysis.py` 和后续分析按控制器类型筛选数据。

## 已实现的控制器

| 类 | `name` | 行为 | 角色 |
|---|---|---|---|
| `ZeroControlController` | `zero_control` | 恒为 0 | 自由漂移基线；协议第 7 节 gate question 1 要求的 u≡0 条件 |
| `RandomExciteController` | `iid` | 每轮独立 Bernoulli(p) 抽样，`p` 和随机种子在构造时传入 | 系统辨识用的开环随机激励；协议第 7 节 gate question 2/3、第 6 节正式采集 |
| `ConstantRemindController` | `constant_remind` | 恒为 1 | "暴力"常控基线 |
| `PeriodicController` | `periodic` | 每 `period` 轮触发一次（轮次从 1 计） | 固定周期重提醒，接近生产环境常见策略 |
| `ThresholdController` | `threshold` | 上一轮 `y_probe` 低于 `y_min` 才触发；无历史或上一轮打分失败（NaN）时不触发 | 经典 bang-bang 反馈基线，是 Koopman-MPC 需要打赢的"简单对手" |
| `KoopmanMPCController` | `koopman_mpc` | 给定拟合好的 `KoopmanSurrogate` + `ReducedStateConfig`，对 0/1 动作空间做短 horizon（默认 2）穷举，用 `surrogate.step()`/`.readout()` 滚动预测每种动作序列下的 y，选第一步动作使预测总分（减 `repeat_penalty`）最高 | 用 Koopman 代理做前瞻规划的执行器；在对抗防御任务 Phase E 四臂闭环验证中打赢 zero_control/threshold，以更低代价追平 constant_remind（见 [../experiments/koopman_defense_pilot.md](../experiments/koopman_defense_pilot.md)）。目前只接入了对抗防御这条线，人格漂移这条线（`screening.py::_make_controller`）还没接 |

`ZeroControlController.name`（`"zero_control"`）和 `RandomExciteController.name`（`"iid"`）
的取值不能改：`analysis.py::analyze_screening` 按这两个精确字符串过滤 `excitation_design`
列来回答协议第 7 节的三个 gate 问题。

## 扩展点

- **把 `KoopmanMPCController` 接入人格漂移这条线**：控制器本身已实现并在对抗防御领域验证过，
  接入 `screening.py::_make_controller` 只需要人格漂移领域自己的拟合好的 `KoopmanSurrogate`
  （等正式 320 条轨迹采集完成后训练），不需要改控制器代码本身。
- **`u_gain`（split-softmax）/ `u_steer`（激活转向）通道**：目前 `next_u_remind` 只覆盖
  channel A。这两个通道需要 hook 模型内部（注意力权重 / 残差流），不是简单的取值选择问题，
  暂不适合塞进同一个接口，留作后续扩展时再设计（见 `../BASELINES.md` 第②档）。
- **`screening.py::_make_controller`** 是目前唯一的 controller 工厂；给新 baseline 接一条
  跑数据的路径时，照它的模式加一个分支或独立的编排函数即可。
