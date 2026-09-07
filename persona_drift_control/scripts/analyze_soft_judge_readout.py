#!/usr/bin/env python3
"""F4 Step 2 (docs/experiments/signal_resolution_plan.md 第五节): the four
RC judges (RC-0/RC-1/RC-2/RC-3, per section 0.5) applied to `y_soft`, the
soft (expected-value) safety judge from F4 Step 1.

"Variable improves" direction: UP (y_soft closer to 1 = safer), same
polarity as the hard label `y_safety`.

RC-1 has no aux dimension for this domain (Phase B has no ERGO-style
`shard_frac`; the exogenous inputs are just `turn` and `u_remind`), so the
three nulls are the direct analogue of the ERGO recipe (section 0.2) with
that substitution: `const` = train global mean; `turn_mean` = train's
per-turn mean; `stateless` = OLS `y_soft ~ 1 + turn + u_remind` fit on train,
evaluated at each held-out row's own turn/u_remind (no y-lag, matching the
ERGO nulls' "no state" property). No rollout recursion needed for the
nulls since they never consume their own past predictions; the ARX model's
`held_out_rollout_mse` comes from the shared `modeling.evaluate.
rollout_output_error` harness unchanged.

CPU-only, pure numpy/scipy. Run directly (no sbatch), after F4 Step 1's
`outputs/koopman_case_study/soft_judge_phaseB.json` exists.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from analyze_readout_state import group_by_key, ols, variance_decomposition  # noqa: E402
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import rollout_output_error  # noqa: E402
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402

HARD_ROWS_PATH = "outputs/koopman_defense_phaseB_random_excite/rejudge_qwen3_4b_instruct_2507/trajectories.jsonl"
SOFT_JUDGE_PATH = "outputs/koopman_case_study/soft_judge_phaseB.json"
OUT_PATH = pathlib.Path("outputs/koopman_case_study/soft_judge_readout_report.json")
N_SPLITS = 20
HELD_OUT_FRAC = 0.25


def key_fn(row: dict) -> str:
    return row["trajectory_id"]


def load_merged_rows() -> list[dict]:
    """Phase B's hard-labeled rows, each augmented with `y_soft` from F4
    Step 1's output (joined on trajectory_id+turn)."""
    hard_rows = load_trajectories(HARD_ROWS_PATH)
    soft = json.loads(pathlib.Path(SOFT_JUDGE_PATH).read_text())
    soft_by_key = {(r["trajectory_id"], r["turn"]): r for r in soft["rows"]}
    merged = []
    for row in hard_rows:
        s = soft_by_key[(row["trajectory_id"], row["turn"])]
        merged.append({**row, "y_soft": s["y_soft"], "argmax_k": s["argmax_k"], "label_mass_total": s["label_mass_total"]})
    return merged


def rc0_report(rows: list[dict]) -> dict:
    y_soft = np.array([r["y_soft"] for r in rows])
    y_hard = np.array([r["y_safety"] for r in rows])
    argmax_k = np.array([r["argmax_k"] for r in rows])
    argmax_score = (argmax_k - 1) / 4.0
    consistency = float(np.mean(np.isclose(argmax_score, y_hard)))
    rho, p = stats.spearmanr(y_soft, y_hard)
    quantiles = np.percentile(y_soft, [0, 25, 50, 75, 100]).tolist()
    hard_value_counts = {
        str(v): int(np.sum(np.isclose(y_hard, v))) for v in sorted(set(np.round(y_hard, 4).tolist()))
    }
    return {
        "argmax_hard_label_consistency": consistency,
        "n_rows": len(rows),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "y_soft_mean": float(y_soft.mean()),
        "y_soft_sd": float(y_soft.std(ddof=1)),
        "y_soft_quantiles_0_25_50_75_100": quantiles,
        "y_hard_value_counts": hard_value_counts,
        "label_mass_total_mean": float(np.mean([r["label_mass_total"] for r in rows])),
    }


def build_gain_transitions(rows: list[dict], x_col: str, include_traj_fe: bool) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    groups = group_by_key(rows, key_fn)
    traj_ids = sorted(groups.keys())
    dummy_ids = traj_ids[:-1] if include_traj_fe else []
    X, y_next, n_reminded = [], [], 0
    for tid, traj in groups.items():
        for a, b in zip(traj, traj[1:]):
            if b["turn"] != a["turn"] + 1:
                continue
            row_x = [1.0, float(a[x_col]), float(a["u_remind"]), float(a["turn"]), float(b["u_remind"])]
            if include_traj_fe:
                row_x.extend(1.0 if tid == other else 0.0 for other in dummy_ids)
            X.append(row_x)
            y_next.append(float(b[x_col]))
            if int(a["u_remind"]) != 0:
                n_reminded += 1
    return np.array(X), np.array(y_next), n_reminded, traj_ids


def rc3_report(rows: list[dict]) -> dict:
    out = {}
    for include_fe, label in ((False, "no_traj_fe"), (True, "with_traj_fe")):
        X, y, n_rem, traj_ids = build_gain_transitions(rows, "y_soft", include_fe)
        gain = ols(X, y)
        out[label] = {
            "u_t_coef": gain["beta"][2],
            "u_t_se": gain["se"][2],
            "u_t_p": gain["p"][2],
            "u_t1_coef": gain["beta"][4],
            "u_t1_se": gain["se"][4],
            "u_t1_p": gain["p"][4],
            "r2": gain["r2"],
            "n_transitions": gain["n"],
            "n_reminded_transitions": n_rem,
            "n_trajectories": len(traj_ids),
        }
    return out


def stateless_null_predict(train_rows: list[dict], eval_rows: list[dict], y_col: str) -> np.ndarray:
    X_train = np.array([[1.0, float(r["turn"]), float(r["u_remind"])] for r in train_rows])
    y_train = np.array([float(r[y_col]) for r in train_rows])
    gain = ols(X_train, y_train)
    beta = np.array(gain["beta"])
    X_eval = np.array([[1.0, float(r["turn"]), float(r["u_remind"])] for r in eval_rows])
    return X_eval @ beta


def rc1_split(rows: list[dict], seed: int, y_col: str) -> dict:
    split = split_by_system_prompt_id(rows, train_frac=1.0 - HELD_OUT_FRAC, val_frac=0.0, seed=seed, split_col="attack_id")
    train_rows, held_out_rows = split["train"], split["test"]

    config = ReducedStateConfig(nu=1, mu=2, contemporaneous_v=True)
    train_dataset = build_identification_dataset(train_rows, config, y_col=y_col)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=1e-6).fit(train_dataset)
    arx_mse = rollout_output_error(model, held_out_rows, config, y_col=y_col)

    # Nulls: predict every held-out row with turn >= 2 (turn 1 has no
    # preceding action/lag for anyone to condition on).
    eval_rows = [r for r in held_out_rows if r["turn"] >= 2]
    y_true = np.array([float(r[y_col]) for r in eval_rows])
    train_eval_rows = [r for r in train_rows if r["turn"] >= 2]

    const_pred = np.full(len(eval_rows), np.mean([float(r[y_col]) for r in train_rows]))
    const_mse = float(np.mean((const_pred - y_true) ** 2))

    turn_mean = {}
    for t in sorted({r["turn"] for r in train_rows}):
        vals = [float(r[y_col]) for r in train_rows if r["turn"] == t]
        turn_mean[t] = float(np.mean(vals)) if vals else float(np.mean([float(r[y_col]) for r in train_rows]))
    turn_mean_pred = np.array([turn_mean.get(r["turn"], np.mean(list(turn_mean.values()))) for r in eval_rows])
    turn_mean_mse = float(np.mean((turn_mean_pred - y_true) ** 2))

    stateless_pred = stateless_null_predict(train_eval_rows, eval_rows, y_col)
    stateless_mse = float(np.mean((stateless_pred - y_true) ** 2))

    best_null = min(const_mse, turn_mean_mse, stateless_mse)
    return {
        "arx_mse": arx_mse,
        "const_mse": const_mse,
        "turn_mean_mse": turn_mean_mse,
        "stateless_mse": stateless_mse,
        "best_null_mse": best_null,
        "beats_best_null": bool(arx_mse < best_null),
    }


def rc1_report(rows: list[dict], y_col: str) -> dict:
    splits = [rc1_split(rows, seed, y_col) for seed in range(N_SPLITS)]
    n_pass = sum(1 for s in splits if s["beats_best_null"])
    return {
        "n_splits": N_SPLITS,
        "splits": splits,
        "arx_mse_mean": float(np.mean([s["arx_mse"] for s in splits])),
        "arx_mse_median": float(np.median([s["arx_mse"] for s in splits])),
        "const_mse_mean": float(np.mean([s["const_mse"] for s in splits])),
        "turn_mean_mse_mean": float(np.mean([s["turn_mean_mse"] for s in splits])),
        "stateless_mse_mean": float(np.mean([s["stateless_mse"] for s in splits])),
        "n_splits_beating_best_null": n_pass,
        "rc1_pass": bool(n_pass >= 18),
    }


def main() -> None:
    rows = load_merged_rows()

    rc0 = rc0_report(rows)
    rc2 = variance_decomposition(rows, y_col="y_soft", key_fn=key_fn)
    hard_lag1 = variance_decomposition(rows, y_col="y_safety", key_fn=key_fn)
    rc3 = rc3_report(rows)
    rc1 = rc1_report(rows, y_col="y_soft")

    report = {
        "direction_convention": "higher y_soft = safer (same polarity as y_safety)",
        "RC0": rc0,
        "RC1": rc1,
        "RC2": rc2,
        "RC2_hard_label_reference": hard_lag1,
        "RC3": rc3,
    }

    rc0_pass = rc0["argmax_hard_label_consistency"] >= 0.95
    rc1_pass = rc1["rc1_pass"]
    rc2_pass = rc2["lag1_demeaned"] > 0 and rc2["lag1_demeaned_p"] < 0.05
    rc3_no_fe = rc3["no_traj_fe"]
    rc3_pass = rc3_no_fe["u_t_coef"] > 0 and rc3_no_fe["u_t_p"] < 0.05
    verdict = {
        "RC0_pass": rc0_pass,
        "RC1_pass": rc1_pass,
        "RC2_pass": rc2_pass,
        "RC3_pass_no_traj_fe": rc3_pass,
        "all_pass": bool(rc0_pass and rc1_pass and rc2_pass and rc3_pass),
    }
    report["verdict"] = verdict

    print("=== F4 Step 2: y_soft against RC-0/1/2/3 ===")
    print(f"G-F4-1 (argmax vs hard label consistency): {rc0['argmax_hard_label_consistency']:.4f} (gate: >= 0.95)")
    print(f"\nRC-0: spearman(y_soft, y_safety_hard) = {rc0['spearman_rho']:.4f} (p={rc0['spearman_p']:.2e})")
    print(f"  y_soft mean={rc0['y_soft_mean']:.4f} sd={rc0['y_soft_sd']:.4f} quantiles={rc0['y_soft_quantiles_0_25_50_75_100']}")
    print(f"  y_hard value counts: {rc0['y_hard_value_counts']}")
    print(f"  -> RC0_pass={rc0_pass}")

    print(f"\nRC-1 (20 splits, held out by attack_id):")
    print(f"  ARX mean={rc1['arx_mse_mean']:.4f} median={rc1['arx_mse_median']:.4f}")
    print(f"  const={rc1['const_mse_mean']:.4f} turn_mean={rc1['turn_mean_mse_mean']:.4f} stateless={rc1['stateless_mse_mean']:.4f}")
    print(f"  splits beating best null: {rc1['n_splits_beating_best_null']}/{N_SPLITS} -> RC1_pass={rc1_pass}")

    print(f"\nRC-2 (de-turned lag-1, y_soft): lag1_raw={rc2['lag1_raw']:.4f} lag1_demeaned={rc2['lag1_demeaned']:.4f} (p={rc2['lag1_demeaned_p']:.2e}) stable_share={rc2['stable_traj_share']*100:.1f}%")
    print(f"  reference (y_safety hard label): lag1_demeaned={hard_lag1['lag1_demeaned']:.4f} (p={hard_lag1['lag1_demeaned_p']:.2e})")
    print(f"  -> RC2_pass={rc2_pass}")

    print("\nRC-3 (input gain, y_soft):")
    for label, r in rc3.items():
        print(
            f"  {label:<14} u_t={r['u_t_coef']:+.4f}(p={r['u_t_p']:.4e})  u_t+1={r['u_t1_coef']:+.4f}(p={r['u_t1_p']:.4e})  "
            f"R2={r['r2']:.3f} n={r['n_transitions']}"
        )
    print(f"  -> RC3_pass (no_traj_fe spec)={rc3_pass}")

    print(f"\n=== verdict: all_pass={verdict['all_pass']} ===")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
