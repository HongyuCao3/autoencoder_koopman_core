#!/usr/bin/env python3
"""docs/next step.md 第一步（零 GPU）：验证"v 槽位错位"诊断的两条可证伪预测。

背景：`ReducedStateConfig.contemporaneous_v`（新增于 dataset.py）把训练对里的
`v` 从"和 z_t 同一个 t"（旧约定，度量的是提醒文本留在历史里对下一轮的残留效应）
改成"z_next 的 y 分量的直接成因"（v=u_(t+1)，匹配 attack_trajectory.py 里
"本轮提醒 -> 本轮回复 -> 本轮打分"这个真实同轮因果链）。见该 dataclass 的
docstring 和 docs/experiments/koopman_defense_pilot.md 的错位分析
（2026-09-02，"docs/next step.md"）。

这个脚本做三件事，对照 docs/next step.md 第一节的两条可证伪预测：

1. 用 Phase B 数据分别拟合旧对齐（contemporaneous_v=False，复现历史上的
   B=-0.059）和新对齐（True）下的 nu=1,mu=1 ARX 模型 -- 预测：新 B 应为正，
   量级接近直接效应。
2. 不依赖任何模型，直接在 Phase B 数据里算 E[y_t|u_t=1]-E[y_t|u_t=0] 作为
   直接效应的模型无关粗估，新 B 应该和它同量级。
3. 用新对齐拟合状态-动作交互模型（modeling.interaction_lift），对 Phase E
   koopman_mpc 臂的真实录制状态做离线重放（沿用
   analyze_state_action_interaction.py 的重放方法，但套一层新对齐的
   ReducedStateConfig/KoopmanMPCController），检查 margin 和 y_probe 的相关
   系数是否从 Phase H 记录的"正相关"（情况越糟边际收益越小，已被证实是
   Phase H 闭环里预算被从最该救的轨迹上抽走的原因）翻转为"负相关"
   （y 越低、提醒的边际收益越大 -- y 有上界 1 带来的饱和效应）。

CPU-only, 纯 numpy -- 不需要 GPU，可以直接跑（不经 sbatch）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.control import KoopmanMPCController  # noqa: E402
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    group_by_trajectory,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import rollout_output_error  # noqa: E402
from persona_drift.modeling.interaction_lift import InteractionLiftedSurrogate, augment_with_interaction  # noqa: E402
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rows-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_defense_phaseB_random_excite/trajectories.jsonl")
    )
    parser.add_argument(
        "--phase-e-koopman-mpc-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/koopman_defense_phaseE_koopman_mpc/trajectories.jsonl"),
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--held-out-frac", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--repeat-penalty", type=float, default=0.0)
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("outputs/koopman_case_study/v_alignment_fix_report.json"))
    return parser.parse_args()


def _fit_arx(train_rows, config, ridge):
    dataset = build_identification_dataset(train_rows, config, y_col="y_safety")
    return KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=ridge).fit(dataset)


def _fit_interaction(train_rows, config, ridge):
    dataset = build_identification_dataset(train_rows, config, y_col="y_safety")
    v_aug = augment_with_interaction(dataset["V"], dataset["Z"], state_index=0)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=ridge).fit({**dataset, "V": v_aug})
    return InteractionLiftedSurrogate(surrogate=model, state_index=0), model


def _replay_phase_e(surrogate, config, horizon, repeat_penalty, phase_e_path):
    controller = KoopmanMPCController(surrogate=surrogate, state_config=config, horizon=horizon, repeat_penalty=repeat_penalty)
    arm_rows = group_by_trajectory(load_trajectories(phase_e_path))
    records = []
    for rows_ in arm_rows.values():
        for i in range(len(rows_)):
            history = rows_[:i]
            z = controller._current_state(history)
            if z is None:
                continue
            value_0 = controller._simulate(z, 0, horizon - 1)
            value_1 = controller._simulate(z, 1, horizon - 1)
            records.append(
                {
                    "trajectory_id": rows_[i].get("trajectory_id"),
                    "turn": rows_[i].get("turn"),
                    "y_probe": float(z[0]),
                    "margin": value_1 - value_0,
                    "action": int(value_1 > value_0),
                }
            )
    return records


def main() -> None:
    args = parse_args()
    rows = load_trajectories(args.rows_path)
    split = split_by_system_prompt_id(
        rows, train_frac=1.0 - args.held_out_frac, val_frac=0.0, seed=args.split_seed, split_col="attack_id"
    )
    train_rows, held_out_rows = split["train"], split["test"]

    # ---- 1. old vs new alignment, nu=1, mu=1 plain ARX ----
    old_config = ReducedStateConfig(nu=1, mu=1, contemporaneous_v=False)
    new_config = ReducedStateConfig(nu=1, mu=1, contemporaneous_v=True)

    old_model = _fit_arx(train_rows, old_config, args.ridge)
    new_model = _fit_arx(train_rows, new_config, args.ridge)

    old_rollout_mse = rollout_output_error(old_model, held_out_rows, old_config, y_col="y_safety")
    new_rollout_mse = rollout_output_error(new_model, held_out_rows, new_config, y_col="y_safety")

    print("=== 1. nu=1, mu=1 ARX: contemporaneous_v=False (旧/残留效应) vs True (新/直接效应) ===")
    print(f"old: B={old_model.B.tolist()}  held_out_rollout_mse={old_rollout_mse:.6f}")
    print(f"new: B={new_model.B.tolist()}  held_out_rollout_mse={new_rollout_mse:.6f}")

    # ---- 2. model-free same-turn contrast on Phase B (train split) ----
    train_y1 = [r["y_safety"] for r in train_rows if float(r["u_remind"]) == 1.0]
    train_y0 = [r["y_safety"] for r in train_rows if float(r["u_remind"]) == 0.0]
    all_y1 = [r["y_safety"] for r in rows if float(r["u_remind"]) == 1.0]
    all_y0 = [r["y_safety"] for r in rows if float(r["u_remind"]) == 0.0]
    contrast_train = float(np.mean(train_y1) - np.mean(train_y0))
    contrast_all = float(np.mean(all_y1) - np.mean(all_y0))

    print("\n=== 2. 模型无关的同轮直接效应粗估: E[y|u=1]-E[y|u=0] ===")
    print(f"train split: n(u=1)={len(train_y1)} n(u=0)={len(train_y0)} contrast={contrast_train:.4f}")
    print(f"all Phase B rows: n(u=1)={len(all_y1)} n(u=0)={len(all_y0)} contrast={contrast_all:.4f}")

    # ---- 2b. same contrast broken out per turn -- pooling across turns can
    # wash out an effect that is real but concentrated in later turns (early
    # turns may sit near y=1 regardless of u_remind, before any attack has
    # had time to erode anything), or reveal that it is genuinely absent at
    # every turn (ruling out "pooling hid it" as the explanation for a small
    # pooled B). ----
    per_turn = {}
    for row in rows:
        per_turn.setdefault(row["turn"], {"y1": [], "y0": []})
        bucket = per_turn[row["turn"]]["y1" if float(row["u_remind"]) == 1.0 else "y0"]
        bucket.append(row["y_safety"])
    print("\n=== 2b. 按轮次拆分的同轮直接效应 (全部 Phase B 行, 未做 train/held-out 切分) ===")
    per_turn_contrast = {}
    for turn in sorted(per_turn):
        y1, y0 = per_turn[turn]["y1"], per_turn[turn]["y0"]
        c = float(np.mean(y1) - np.mean(y0)) if y1 and y0 else None
        per_turn_contrast[turn] = {"n_u1": len(y1), "n_u0": len(y0), "contrast": c}
        c_str = f"{c:.4f}" if c is not None else "n/a"
        print(f"turn={turn}: n(u=1)={len(y1)} n(u=0)={len(y0)} contrast={c_str}")

    # ---- 3. interaction model under new alignment + Phase E replay ----
    new_interaction, new_interaction_raw = _fit_interaction(train_rows, new_config, args.ridge)
    print("\n=== 3. 新对齐下的交互模型 (nu=1,mu=1) ===")
    print(f"B={new_interaction_raw.B.tolist()}  (第2列是交互项系数 v*y)")

    records = _replay_phase_e(new_interaction, new_config, args.horizon, args.repeat_penalty, args.phase_e_koopman_mpc_path)
    margins = np.array([r["margin"] for r in records])
    y_probes = np.array([r["y_probe"] for r in records])
    actions = np.array([r["action"] for r in records])
    corr = float(np.corrcoef(margins, y_probes)[0, 1]) if len(records) > 1 and margins.std() > 0 else None

    print(f"\nPhase E koopman_mpc 状态离线重放: n={len(records)}")
    print(f"margin std={margins.std():.6f} min={margins.min():.4f} max={margins.max():.4f}")
    print(f"n_remind={int(actions.sum())}/{len(actions)} (genuine mix: {0 < actions.sum() < len(actions)})")
    print(f"corr(margin, y_probe)={corr}")
    print("预测：corr 应为负（y 越低边际收益越大）-- Phase H 记录的是正相关（错误方向）。")

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "arx_old": {"B": old_model.B.tolist(), "A": old_model.A.tolist(), "b": old_model.b.tolist(), "held_out_rollout_mse": old_rollout_mse},
        "arx_new": {"B": new_model.B.tolist(), "A": new_model.A.tolist(), "b": new_model.b.tolist(), "held_out_rollout_mse": new_rollout_mse},
        "model_free_contrast": {
            "train_split": {"n_u1": len(train_y1), "n_u0": len(train_y0), "contrast": contrast_train},
            "all_rows": {"n_u1": len(all_y1), "n_u0": len(all_y0), "contrast": contrast_all},
            "per_turn": per_turn_contrast,
        },
        "interaction_new": {"B": new_interaction_raw.B.tolist()},
        "phase_e_replay": {
            "n_records": len(records),
            "margin_std": float(margins.std()) if len(records) else None,
            "margin_min": float(margins.min()) if len(records) else None,
            "margin_max": float(margins.max()) if len(records) else None,
            "n_remind": int(actions.sum()) if len(records) else None,
            "corr_margin_y_probe": corr,
            "records": records,
        },
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
