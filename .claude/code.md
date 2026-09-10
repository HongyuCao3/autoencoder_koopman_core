# 代码规则

先读 `.claude/global.md`。

## 反模式（硬约束）

- **不要改** `control.py` / `controller_cli.py` / `modeling/evaluate.py`——它们是已提交结果的
  复现路径。需要新行为就加新参数/新模块，默认值保持旧行为，并跑回归检验证明逐字节相同
  （参照 K2.1 的 `--n-folds` 补丁做法）。
- **不要覆盖任何已有产物。** 输出路径一律新建。
- **守卫与其元测试不得删除、不得绕过。** 已有：`run_config_guard`（防跨臂产物混合）、
  `state_action_interaction_guard`。让某次运行通过的正确做法是改配置，不是改守卫。
- 报告口径类的检查要写成**运行前 raise 的守卫**，不是文档里的一句话。

## 风格

- 函数保持单一职责；只用一次的东西不要抽象成 helper。
- 不给未改动的代码补 docstring / 注释。
- 不写向后兼容 shim；无用代码直接删。
  **历史包名例外（2026-09-10 用户裁定）**：目录名 `persona_drift_control/` 与包名
  `persona_drift` 与当前任务早已脱钩，但**不改名、也不写 alias shim**——`outputs/` 下 50+ 份
  已完成实验的 `.hydra/hydra.yaml` 记着 `persona_drift.*` 的 `_target_`，改名就让论文证据
  盘点要读的历史产物无法重放。理由与规模见 `docs/DOC_CLEANUP_PLAN.md` §五。
  这条例外**只覆盖这一个标识符**，不是"可以留 shim"的一般许可。
- 只在系统边界做校验（CLI 入参、外部 API、产物读入）。

## 测试

```bash
export PATH=/scratch/hcao2/envs/persona_drift_pilot/bin:$PATH
cd persona_drift_control && pytest tests/ -q     # 65 个测试文件
pytest tests/ -q   # core
```

新增行为必须带测试。改动涉及已提交结果的复现路径时，测试要证明**旧路径逐字节不变**。
