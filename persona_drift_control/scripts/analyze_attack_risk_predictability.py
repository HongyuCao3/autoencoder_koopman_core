#!/usr/bin/env python3
"""A1 (docs/experiments/defense_line_redesign_plan.md section 2): can the danger
of a multi-turn attack be predicted from the attack text alone, before any reply
is observed? Cross-trajectory budget allocation (A2) only makes sense if the
answer is yes -- otherwise route A is dead and the defense line is left with
route B (drop the 256-token truncation, B1) or a negative result.

CPU-only, pure numpy/scipy (no sklearn in this env; ridge regression has a
closed form so no dependency is needed). Population and per-trajectory late_y
follow section 0.1's construction exactly (verified to reproduce n=255
trajectories / 30 attacks, see REPRODUCTION NOTE below).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

import numpy as np

STOPWORDS = set(
    """
    a an the of to in on for and or with without how what who was were is are be been being
    by at from as that this these those it its his her their you your i we they can could would should
    do does did done make making made use used using give given get getting about into over under more
    most some any all not no if then than there here when where which while up out off very just also
    me my him them he she
    """.split()
)

CONTENT_WORD_RE = re.compile(r"[a-z]{3,}")
REJUDGE_SUBDIR = "rejudge_qwen3_4b_instruct_2507"
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
ACTOR_CARDINALITY_LIMIT = 8


def content_words(text: str) -> list[str]:
    return [w for w in CONTENT_WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def target_hit_rate(goal_text: str, target_text: str) -> float:
    words = sorted(set(content_words(goal_text)))
    if not words:
        return 0.0
    target_low = target_text.lower()
    hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", target_low))
    return hits / len(words)


def load_pool(root: pathlib.Path) -> list[dict]:
    arm_dirs = [
        d
        for d in root.iterdir()
        if d.is_dir()
        and (d / REJUDGE_SUBDIR / "trajectories.jsonl").exists()
        and not d.name.endswith("_indepjudge")
    ]
    rows = []
    for d in sorted(arm_dirs):
        with open(d / REJUDGE_SUBDIR / "trajectories.jsonl") as f:
            for line in f:
                r = json.loads(line)
                r["_arm"] = d.name
                rows.append(r)
    return rows


def build_trajectory_table(rows: list[dict]) -> dict:
    traj = defaultdict(list)
    for r in rows:
        traj[(r["_arm"], r["trajectory_id"])].append(r)

    def early_reminder(vs):
        return any(r["turn"] in (1, 2) and r.get("u_remind") == 1 for r in vs)

    kept = {k: v for k, v in traj.items() if not early_reminder(v)}
    data = {}
    for k, vs in kept.items():
        turn1 = next(r for r in vs if r["turn"] == 1)
        data[k] = dict(
            attack_id=vs[0]["attack_id"],
            category=vs[0]["category"],
            actor_name=vs[0]["actor_name"],
            plain_query=vs[0]["plain_query"],
            attacker_query_t1=turn1["attacker_query"],
            late_self=float(np.mean([r["y_safety_self_judge"] for r in vs if r["turn"] in (3, 4, 5)])),
            late_indep=float(np.mean([r["y_safety"] for r in vs if r["turn"] in (3, 4, 5)])),
        )
    return data


def build_features(data: dict) -> tuple[np.ndarray, list[str], list[str]]:
    keys = list(data.keys())
    categories = sorted(set(data[k]["category"] for k in keys))
    actor_names = sorted(set(data[k]["actor_name"] for k in keys))
    use_actor = len(actor_names) <= ACTOR_CARDINALITY_LIMIT

    feature_names = [f"category={c}" for c in categories]
    feature_names += ["plain_query_wordcount", "plain_query_contentcount", "t1_query_wordcount", "t1_target_hit_rate"]
    if use_actor:
        feature_names += [f"actor={a}" for a in actor_names]

    rows = []
    for k in keys:
        d = data[k]
        cat_oh = [1.0 if d["category"] == c else 0.0 for c in categories]
        pq_words = d["plain_query"].split()
        pq_content = content_words(d["plain_query"])
        t1_words = d["attacker_query_t1"].split()
        hit_rate = target_hit_rate(d["plain_query"], d["attacker_query_t1"])
        feats = cat_oh + [len(pq_words), len(pq_content), len(t1_words), hit_rate]
        if use_actor:
            feats += [1.0 if d["actor_name"] == a else 0.0 for a in actor_names]
        rows.append(feats)
    X = np.array(rows, dtype=float)
    return X, feature_names, keys


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    x_std[x_std == 0] = 1.0
    Xs = (X - x_mean) / x_std
    y_mean = y.mean()
    ys = y - y_mean
    n_features = Xs.shape[1]
    A = Xs.T @ Xs + alpha * np.eye(n_features)
    b = Xs.T @ ys
    beta = np.linalg.solve(A, b)
    return beta, x_mean, x_std, y_mean


def ridge_predict(beta, x_mean, x_std, y_mean, X: np.ndarray) -> np.ndarray:
    Xs = (X - x_mean) / x_std
    return Xs @ beta + y_mean


def r2_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sst = np.sum((y_true - y_true.mean()) ** 2)
    if sst == 0:
        return float("nan")
    sse = np.sum((y_true - y_pred) ** 2)
    return 1.0 - sse / sst


def select_alpha(X: np.ndarray, y: np.ndarray, attack_ids: np.ndarray, rng: np.random.Generator) -> float:
    """5-fold CV over attacks (not rows) so same-attack trajectories never split
    across train/val -- avoids leaking attack identity into alpha selection."""
    unique_attacks = np.unique(attack_ids)
    perm = rng.permutation(len(unique_attacks))
    folds = np.array_split(perm, 5)
    best_alpha, best_score = ALPHA_GRID[0], -np.inf
    for alpha in ALPHA_GRID:
        preds = np.full(len(y), np.nan)
        for fold in folds:
            val_attacks = set(unique_attacks[fold])
            val_mask = np.array([a in val_attacks for a in attack_ids])
            if val_mask.sum() == 0 or (~val_mask).sum() == 0:
                continue
            beta, x_mean, x_std, y_mean = ridge_fit(X[~val_mask], y[~val_mask], alpha)
            preds[val_mask] = ridge_predict(beta, x_mean, x_std, y_mean, X[val_mask])
        score = r2_pooled(y[~np.isnan(preds)], preds[~np.isnan(preds)])
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


def leave_one_attack_out(X: np.ndarray, y: np.ndarray, attack_ids: np.ndarray, rng: np.random.Generator):
    unique_attacks = np.unique(attack_ids)
    preds = np.full(len(y), np.nan)
    chosen_alphas = {}
    for held_out in unique_attacks:
        test_mask = attack_ids == held_out
        train_mask = ~test_mask
        alpha = select_alpha(X[train_mask], y[train_mask], attack_ids[train_mask], rng)
        chosen_alphas[held_out] = alpha
        beta, x_mean, x_std, y_mean = ridge_fit(X[train_mask], y[train_mask], alpha)
        preds[test_mask] = ridge_predict(beta, x_mean, x_std, y_mean, X[test_mask])
    return preds, chosen_alphas


def bootstrap_r2_by_attack(y: np.ndarray, preds: np.ndarray, attack_ids: np.ndarray, n_boot: int, seed: int):
    unique_attacks = np.unique(attack_ids)
    rng = np.random.default_rng(seed)
    idx_by_attack = {a: np.where(attack_ids == a)[0] for a in unique_attacks}
    boot_r2 = []
    for _ in range(n_boot):
        sample_attacks = rng.choice(unique_attacks, size=len(unique_attacks), replace=True)
        idx = np.concatenate([idx_by_attack[a] for a in sample_attacks])
        boot_r2.append(r2_pooled(y[idx], preds[idx]))
    boot_r2 = np.array(boot_r2)
    point = r2_pooled(y, preds)
    lo, hi = np.percentile(boot_r2, [2.5, 97.5])
    return point, (float(lo), float(hi))


def oracle_attack_mean_r2(y: np.ndarray, attack_ids: np.ndarray) -> float:
    """Section 0.1's ceiling: predict each trajectory with the mean late_y of the
    OTHER trajectories sharing its attack_id (leave-one-trajectory-out, not
    leave-one-attack-out -- this needs replicate trajectories per attack, which
    a genuinely novel attack at deployment time would not have)."""
    preds = np.full(len(y), np.nan)
    for a in np.unique(attack_ids):
        mask = attack_ids == a
        vals = y[mask]
        n = len(vals)
        if n > 1:
            preds[mask] = (vals.sum() - vals) / (n - 1)
        else:
            preds[mask] = vals.mean()
    return r2_pooled(y, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", type=pathlib.Path, default=pathlib.Path("outputs"))
    parser.add_argument("--report-path", type=pathlib.Path, default=pathlib.Path("outputs/attack_risk_predictability_report.json"))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    rows = load_pool(args.outputs_root)
    assert len(rows) == 1905, f"expected 1905 pooled rows, got {len(rows)} -- population definition has drifted, stop"

    data = build_trajectory_table(rows)
    assert len(data) == 255, f"expected 255 kept trajectories, got {len(data)}"
    n_attacks = len(set(d["attack_id"] for d in data.values()))
    assert n_attacks == 30, f"expected 30 attacks, got {n_attacks}"

    X, feature_names, keys = build_features(data)
    attack_ids = np.array([data[k]["attack_id"] for k in keys])

    report = {
        "gate_G_A1_0_population": {"n_rows_pool": len(rows), "n_trajectories": len(data), "n_attacks": n_attacks, "passed": True},
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_categories_found": len(set(d["category"] for d in data.values())),
        "categories_found": sorted(set(d["category"] for d in data.values())),
        "actor_cardinality": len(set(d["actor_name"] for d in data.values())),
    }

    for col in ("late_self", "late_indep"):
        y = np.array([data[k][col] for k in keys], dtype=float)
        rng = np.random.default_rng(0)

        oracle_r2 = oracle_attack_mean_r2(y, attack_ids)

        preds, chosen_alphas = leave_one_attack_out(X, y, attack_ids, rng)
        loao_point, loao_ci = bootstrap_r2_by_attack(y, preds, attack_ids, args.n_boot, seed=0)

        report[col] = {
            "oracle_attack_mean_r2_recomputed": round(float(oracle_r2), 3),
            "loao_ridge_r2_point": round(float(loao_point), 4),
            "loao_ridge_r2_ci95": [round(loao_ci[0], 4), round(loao_ci[1], 4)],
            "global_mean_r2_baseline": 0.0,
            "chosen_alphas_by_held_out_attack": {a: chosen_alphas[a] for a in attack_ids},
        }

    documented = {"late_self": 0.571, "late_indep": 0.453}
    for col, doc_val in documented.items():
        recomputed = report[col]["oracle_attack_mean_r2_recomputed"]
        report[col]["gate_G_A1_1_documented_r2"] = doc_val
        report[col]["gate_G_A1_1_matches_2dp"] = round(recomputed, 2) == round(doc_val, 2)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
