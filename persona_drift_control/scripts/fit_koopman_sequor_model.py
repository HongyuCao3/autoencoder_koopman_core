#!/usr/bin/env python3
"""S2: fit `y_(t+1) = A y_t + B u_t + c` on the `constraint` line's S1b
excitation arms (docs/experiments/constraint_retention_plan.md section 5).
CPU only, pure numpy -- run directly, no sbatch.

    G-S2-1 state        one-step scalar-y MSE beats the best of three trivial
                        nulls (`const` / `turn_mean` / `stateless`) on >= 14
                        of 20 item-disjoint folds. Every null is handed
                        `turn`, the deterministic exogenous quantity the
                        surrogate also has -- denying it would manufacture a
                        pass (the ERGO RC-1 defect, mirrored).
    G-S2-2 controllable `B`'s 95% CI excludes 0 with the sign that says a
                        reminder raises `y`.
    G-S2-3 degeneracy   spectral radius in (0.1, 1.05), full controllability
                        rank, Gramian condition < 1e12. RECORDED, not a gate.
    G-S2-4 sizing       trajectories per arm S3 would need, from this fit's
                        `B` and the late-window paired sd. > 150 per arm ->
                        stop and report.

The input is S1b (`bernoulli` + `antithetic`) and only S1b: the S1a arms are a
full-dose endpoint contrast with `u` constant within an arm, which identifies
no `B` at all. `--contemporaneous-v` is mandatory, not default -- the reminder
for turn t is injected BEFORE that turn's reply is generated and scored, so
`u_t` acts on `y_t`, and fitting the lagged slot would estimate the carryover
of last turn's reminder text while calling it the actuator's effect. Both the
`defense` line and ERGO retracted a verdict over exactly that off-by-one.

WHAT A WEAK RESULT HERE DOES NOT MEAN (screening section 10 item 9, signed
2026-09-10): 65% of the judged constraints on this item set lie outside the
judge's calibration set, and that observation noise biases the state
coefficient toward zero. A weak or null G-S2-1 on these rows therefore has two
possible causes this design cannot separate, and it does NOT close the line.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    group_by_trajectory,
)
from persona_drift.modeling.koopman import (  # noqa: E402
    KoopmanSurrogate,
    abs_sign_extra_features,
    no_extra_features,
)

S1B_ARMS = ("bernoulli", "antithetic")
N_BOOTSTRAP = 10000
POWER_Z = 2.80  # two-sided alpha=0.05 at 80% power, as in every gate on this line
S3_TRAJECTORIES_PER_ARM_CEILING = 150  # plan section 5, G-S2-4


def _sibling_module(name: str):
    """Import the sibling script that already defines a criterion rather than
    restating it. The three nulls come from the `defense` line's fit script,
    where G-K2-1 used them: two copies of a null definition drift apart, and
    the whole point of G-S2-1 is that it is the same comparison K2 ran."""

    path = pathlib.Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--independent-readout", type=pathlib.Path, required=True,
                   help="score_sequor_trajectories.py report with judge_kind=independent.")
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="Must not already exist.")
    p.add_argument("--nu", type=int, default=1)
    p.add_argument("--mu", type=int, default=1)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--n-folds", type=int, default=20, help="Item-disjoint folds for G-S2-1.")
    p.add_argument("--folds-to-pass", type=int, default=14, help="Plan section 5: 14 of 20.")
    p.add_argument("--controllability-horizon", type=int, default=5)
    p.add_argument("--late-from", type=int, default=15, help="Late window used by G-S2-4's sd.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-item", action="append", default=[], metavar="ITEM_ID")
    p.add_argument("--contemporaneous-v", action="store_true", required=True,
                   help="Mandatory on this line. `u_t` is injected before turn t's reply is "
                        "generated, so it acts within the turn; the lagged slot would estimate "
                        "carryover instead of the actuator.")
    return p.parse_args()


def rows_for_identification(readout: dict, excluded: list[str]) -> list[dict]:
    """S1b rows in the shape `modeling.dataset` expects.

    An unusable readout (the judge's verdict did not parse) becomes NaN rather
    than being dropped here: `build_reduced_state_pairs` drops every pair that
    would touch it, which is the difference between losing one transition and
    silently splicing turn t-1 onto turn t+1 as if they were adjacent.
    """

    rows = []
    for row in readout["rows"]:
        if row["branch"] not in S1B_ARMS or row["item_id"] in excluded:
            continue
        rows.append({
            "trajectory_id": row["trajectory_id"], "item_id": row["item_id"],
            "turn": row["turn"], "seed": row["seed"],
            "y_graded": float("nan") if row["y_graded"] is None else float(row["y_graded"]),
            "u_remind": float(row["u_remind"]),
        })
    return rows


def design_matrix(rows: list[dict], config: ReducedStateConfig) -> dict:
    """The same transitions `build_identification_dataset` produces, tagged
    with the item each one came from, as an explicit regression design.

    The design is `[z_t, v_t, 1]` with target `y_(t+1)` -- exactly the column
    of `KoopmanSurrogate.fit`'s ridge solve that carries `y` when the lift is
    the identity. It exists so `B` can be bootstrapped over items in closed
    form (per-item Gram matrices summed per resample) instead of refitting the
    surrogate ten thousand times. `_assert_matches_surrogate` is what keeps
    the two from drifting: if they ever disagree, the CI would belong to a
    model nobody fit.

    The transitions come from the builder itself, one item at a time, rather
    than from a hand-rolled copy of its loop -- including its NaN filter,
    which drops every pair touching an unparsed verdict. A private copy of
    that loop would drift from the builder the first time either changed, and
    the drift would be silent.
    """

    blocks, targets, items = [], [], []
    by_item: dict[str, list[dict]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
    for item in sorted(by_item):
        data = build_identification_dataset(by_item[item], config, y_col="y_graded", u_col="u_remind")
        if data["Z"].shape[0] == 0:
            continue
        n = data["Z"].shape[0]
        blocks.append(np.hstack([data["Z"], data["V"], np.ones((n, 1))]))
        targets.append(data["Z_next"][:, config.nu - 1])
        items.extend([item] * n)
    return {
        "X": np.vstack(blocks), "y": np.concatenate(targets), "items": np.array(items),
        "u_index": config.nu + config.mu,  # z = [y lags, u lags]; v sits right after
    }


def _ridge_solve(gram: np.ndarray, moment: np.ndarray, ridge: float) -> np.ndarray:
    return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), moment)


def _assert_matches_surrogate(design: dict, dataset: dict, ridge: float) -> float:
    """The explicit solve must reproduce the surrogate's own `B` exactly."""

    beta = _ridge_solve(design["X"].T @ design["X"], design["X"].T @ design["y"], ridge)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=ridge).fit(dataset)
    explicit = float(beta[design["u_index"]])
    fitted = float(model.B[0, 0])
    if not math.isclose(explicit, fitted, rel_tol=1e-8, abs_tol=1e-10):
        raise SystemExit(
            f"explicit design gives B={explicit:.8f} but KoopmanSurrogate gives {fitted:.8f}: "
            f"the bootstrap would describe a model nobody fit")
    return fitted


def bootstrap_b(design: dict, ridge: float, seed: int) -> dict:
    """Resample ITEMS with replacement. A dialogue is the independent unit --
    turns inside one are dependent, and three seeds of one item replay the
    same constraints against the same user turns."""

    items = sorted(set(design["items"].tolist()))
    per_item = {}
    for item in items:
        mask = design["items"] == item
        x, y = design["X"][mask], design["y"][mask]
        per_item[item] = (x.T @ x, x.T @ y)

    rng = np.random.default_rng(seed)
    idx = design["u_index"]
    draws = []
    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(len(items), size=len(items), replace=True)
        gram = sum(per_item[items[i]][0] for i in chosen)
        moment = sum(per_item[items[i]][1] for i in chosen)
        draws.append(float(_ridge_solve(gram, moment, ridge)[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    point = float(_ridge_solve(design["X"].T @ design["X"], design["X"].T @ design["y"], ridge)[idx])
    return {"B": point, "ci": [float(lo), float(hi)], "bootstrap_sd": float(np.std(draws, ddof=1)),
            "n_items": len(items), "n_transitions": int(design["X"].shape[0])}


def fold_evaluation(defense, rows: list[dict], config: ReducedStateConfig, ridge: float,
                    n_folds: int, seed: int) -> dict:
    """Item-disjoint folds: the surrogate's scalar-y one-step MSE against the
    three nulls, on the identical transition set.

    Splitting by item (not by turn, not by trajectory) is the same anti-leak
    rule the `defense` line used for `attack_id`: turns within one dialogue
    are dependent, and the three seeds of one item share its constraints.
    """

    items = sorted({row["item_id"] for row in rows})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(items))
    groups = [shuffled[i::n_folds] for i in range(n_folds)]

    folds = []
    for held_out in groups:
        if not held_out:
            continue
        held = set(held_out)
        train_rows = [r for r in rows if r["item_id"] not in held]
        test_rows = [r for r in rows if r["item_id"] in held]
        train_dataset = build_identification_dataset(train_rows, config, y_col="y_graded", u_col="u_remind")
        test_dataset = build_identification_dataset(test_rows, config, y_col="y_graded", u_col="u_remind")
        if train_dataset["Z"].shape[0] == 0 or test_dataset["Z"].shape[0] == 0:
            continue

        keys = ("y_now", "v", "y_next", "turn_next")
        train_aligned = dict(zip(keys, defense._aligned_transitions(train_rows, config, y_col="y_graded")))
        test_aligned = dict(zip(keys, defense._aligned_transitions(test_rows, config, y_col="y_graded")))
        assert len(test_aligned["y_next"]) == test_dataset["Z"].shape[0], "aux-turn alignment drifted"

        fold = {"held_out_items": sorted(held),
                "n_train_transitions": int(train_dataset["Z"].shape[0]),
                "n_test_transitions": int(len(test_aligned["y_next"]))}
        for name, extra in (("arx", no_extra_features), ("richer_abs_sign", abs_sign_extra_features)):
            model = KoopmanSurrogate(extra_features_fn=extra, ridge=ridge).fit(train_dataset)
            pred = np.array([model.step(z, v)[config.nu - 1]
                             for z, v in zip(test_dataset["Z"], test_dataset["V"])])
            fold[f"{name}_y_one_step_mse"] = float(np.mean((pred - test_aligned["y_next"]) ** 2))
        for name, pred in defense._null_predictions(train_aligned, test_aligned).items():
            fold[f"null_{name}_y_one_step_mse"] = float(np.mean((pred - test_aligned["y_next"]) ** 2))
        fold["best_null_y_one_step_mse"] = min(
            fold[f"null_{name}_y_one_step_mse"] for name in ("const", "turn_mean", "stateless"))
        for name in ("arx", "richer_abs_sign"):
            fold[f"{name}_beats_best_null"] = bool(
                fold[f"{name}_y_one_step_mse"] < fold["best_null_y_one_step_mse"])
        folds.append(fold)

    summary = {"n_folds": len(folds), "folds": folds}
    for name in ("arx", "richer_abs_sign"):
        summary[f"{name}_n_folds_beating_best_null"] = sum(1 for f in folds if f[f"{name}_beats_best_null"])
        summary[f"{name}_mean_y_one_step_mse"] = float(np.mean([f[f"{name}_y_one_step_mse"] for f in folds]))
    for name in ("const", "turn_mean", "stateless"):
        summary[f"null_{name}_mean_y_one_step_mse"] = float(
            np.mean([f[f"null_{name}_y_one_step_mse"] for f in folds]))
    return summary


def gate_s2_1(folds: dict, folds_to_pass: int, n_folds: int) -> dict:
    won = folds["arx_n_folds_beating_best_null"]
    return {
        "criterion": f"arx one-step scalar-y MSE beats the best trivial null on >= {folds_to_pass} "
                     f"of {n_folds} item-disjoint folds; every null gets `turn`",
        "arx_folds_beating_best_null": won, "n_folds": folds["n_folds"],
        "richer_abs_sign_folds_beating_best_null": folds["richer_abs_sign_n_folds_beating_best_null"],
        "arx_mean_mse": folds["arx_mean_y_one_step_mse"],
        "best_null_mean_mse": min(folds[f"null_{n}_mean_y_one_step_mse"]
                                  for n in ("const", "turn_mean", "stateless")),
        "verdict": "PASS" if won >= folds_to_pass else "FAIL",
        "a_weak_result_does_not_close_the_line": (
            "65% of judged constraints sit outside the judge's calibration set; that observation "
            "noise biases the state coefficient toward zero. A weak result here has two causes "
            "this design cannot separate (screening section 10 item 9) -- report and hand back."),
    }


def gate_s2_2(b: dict) -> dict:
    excludes_zero = bool(b["ci"][0] * b["ci"][1] > 0)
    mde = POWER_Z * b["bootstrap_sd"]
    return {
        "criterion": "B's 95% CI excludes 0, sign positive (a reminder raises y)",
        **b, "mde_at_80pct": mde, "effect_over_mde": abs(b["B"]) / mde if mde else None,
        "ci_excludes_zero": excludes_zero, "sign_is_positive": b["B"] > 0,
        "verdict": "PASS" if (excludes_zero and b["B"] > 0) else "FAIL",
        "caveat": None if abs(b["B"]) >= mde else (
            "significant at ~1.96 sigma but below the 80%-power MDE: the point estimate is "
            "likely inflated, as with S0-0's K3 under the old harness"),
    }


def gate_s2_3(diagnostics: dict, state_dim: int) -> dict:
    radius = diagnostics["spectral_radius"]
    return {
        "criterion": "recorded, not a gate: spectral radius in (0.1, 1.05), full controllability "
                     "rank, Gramian condition < 1e12",
        "spectral_radius": radius, "controllability_rank": diagnostics["controllability_rank"],
        "state_dim": state_dim, "gramian_condition": diagnostics["gramian_condition"],
        "radius_in_band": bool(0.1 < radius < 1.05),
        "full_rank": diagnostics["controllability_rank"] == state_dim,
        "gramian_well_conditioned": diagnostics["gramian_condition"] < 1e12,
    }


def gate_s2_4(b: float, late_paired_sd: float, effects: dict) -> dict:
    """Trajectories per arm S3 needs, at 80% power, for each candidate effect.

    The plan says to back it out "from `B` and the late-window paired sd" but
    does not say which contrast S3 must resolve -- and under an equal-budget
    comparison the arms differ in the PLACEMENT of reminders, not their
    number, so `B` alone does not name that effect. The table is therefore
    computed for every effect size the design can name, and the choice is
    marked as a ruling rather than guessed: picking one here would be setting
    S3's power target from whichever number happens to look affordable.
    """

    per_arm = {
        name: math.ceil((POWER_Z * late_paired_sd / effect) ** 2) if effect else None
        for name, effect in effects.items()
    }
    affordable = {name: (n is not None and n <= S3_TRAJECTORIES_PER_ARM_CEILING)
                  for name, n in per_arm.items()}
    return {
        "criterion": f"trajectories per arm at 80% power; > {S3_TRAJECTORIES_PER_ARM_CEILING} per "
                     f"arm -> stop and report",
        "late_window_paired_sd": late_paired_sd, "B": b,
        "candidate_effects": effects, "trajectories_per_arm": per_arm,
        "within_ceiling": affordable,
        "needs_a_ruling": "which candidate effect S3 must resolve. Under an equal-budget design "
                          "the arms differ in reminder PLACEMENT, not count, so B does not name "
                          "the contrast; the plan leaves it open and this script will not pick.",
        "verdict": "NEEDS_RULING",
    }


def state_provenance(rows: list[dict]) -> dict:
    """DIAGNOSTIC, not a gate: where the state term's predictive power comes
    from -- turn-to-turn dynamics, or a level that never changes.

    `y_t` can predict `y_(t+1)` for two very different reasons: the dialogue
    moves and the operator tracks the movement, or this item is simply harder
    than the others and both turns inherit the same level. The fold nulls
    cannot separate them (a held-out item's own mean is not available at test
    time, so it is not a legitimate null), but the same regression run on
    demeaned data can say how much survives.

    Three specifications of `y_(t+1) ~ y_t + u + turn`: pooled, with item
    means removed, and with each trajectory's own mean removed. The last is
    the harshest and is downward biased at fixed T (Nickell, order
    -(1+rho)/(T-1)), so it is a lower bound, not the estimate. The `u`
    coefficient is printed in all three because an actuator effect that moves
    with the specification would be a level artifact rather than an action.
    """

    def fit(group_key: str | None) -> dict:
        y0 = np.array([r["y_now"] for r in transitions])
        y1 = np.array([r["y_next"] for r in transitions])
        u = np.array([r["u"] for r in transitions])
        turn = np.array([r["turn"] for r in transitions])
        if group_key is not None:
            keys = np.array([r[group_key] for r in transitions])
            for key in set(keys.tolist()):
                mask = keys == key
                y0[mask] -= y0[mask].mean()
                y1[mask] -= y1[mask].mean()
                u[mask] -= u[mask].mean()
        design = np.column_stack([np.ones_like(y0), y0, u, turn - turn.mean()])
        beta = np.linalg.lstsq(design, y1, rcond=None)[0]
        return {"y_prev_coefficient": float(beta[1]), "u_coefficient": float(beta[2])}

    transitions = []
    for traj_rows in group_by_trajectory(rows).values():
        ordered = sorted(traj_rows, key=lambda r: r["turn"])
        for now, nxt in zip(ordered, ordered[1:]):
            if now["y_graded"] != now["y_graded"] or nxt["y_graded"] != nxt["y_graded"]:
                continue
            transitions.append({
                "y_now": now["y_graded"], "y_next": nxt["y_graded"], "u": nxt["u_remind"],
                "turn": float(nxt["turn"]), "item_id": now["item_id"],
                "trajectory_id": now["trajectory_id"],
            })
    return {
        "is_a_gate": False,
        "n_transitions": len(transitions),
        "pooled": fit(None),
        "item_demeaned": fit("item_id"),
        "trajectory_demeaned": fit("trajectory_id"),
        "reading": "the gap between pooled and item-demeaned is the share of the state term "
                   "carried by persistent per-item difficulty rather than turn-to-turn dynamics; "
                   "the trajectory-demeaned row is a downward-biased lower bound at fixed T",
    }


def late_window_paired_sd(readout: dict, excluded: list[str], late_from: int) -> float:
    """sd across (item, seed) cells of the late-window mean `y`, pooled over
    the S1b arms -- the unit S3's paired comparison will be made on."""

    cells: dict[tuple[str, str, int], list[float]] = {}
    for row in readout["rows"]:
        if row["branch"] not in S1B_ARMS or row["item_id"] in excluded:
            continue
        if row["y_graded"] is None or row["turn"] < late_from:
            continue
        cells.setdefault((row["branch"], row["item_id"], row["seed"]), []).append(row["y_graded"])
    means = [statistics.fmean(v) for v in cells.values()]
    return statistics.stdev(means)


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    gates = _sibling_module("analyze_sequor_s0_0_gates.py")
    defense = _sibling_module("fit_koopman_defense_model.py")

    readout = gates.load_readout(args.independent_readout, "independent")
    arm_dir = pathlib.Path(readout["arm_dir"])
    arm_report = json.loads((arm_dir / "arm_report.json").read_text())
    cap_record = gates.refuse_if_the_cap_bound(arm_report, args.exclude_item)

    config = ReducedStateConfig(nu=args.nu, mu=args.mu, contemporaneous_v=args.contemporaneous_v)
    rows = rows_for_identification(readout, args.exclude_item)
    if not rows:
        raise SystemExit("no S1b rows: this readout has no bernoulli/antithetic arm")
    dataset = build_identification_dataset(rows, config, y_col="y_graded", u_col="u_remind")
    design = design_matrix(rows, config)
    if design["X"].shape[0] != dataset["Z"].shape[0]:
        raise SystemExit(f"explicit design has {design['X'].shape[0]} transitions but the "
                         f"identification dataset has {dataset['Z'].shape[0]}")

    _assert_matches_surrogate(design, dataset, args.ridge)
    model = KoopmanSurrogate(extra_features_fn=no_extra_features, ridge=args.ridge).fit(dataset)
    diagnostics = model.controllability(args.controllability_horizon)

    folds = fold_evaluation(defense, rows, config, args.ridge, args.n_folds, args.seed)
    b = bootstrap_b(design, args.ridge, args.seed)
    sd = late_window_paired_sd(readout, args.exclude_item, args.late_from)
    s1a_late = 0.1389  # the S1a contrast on this arm, the full-dose ceiling any schedule can buy

    report = {
        "arm_dir": str(arm_dir), "provenance": readout.get("provenance"),
        "judge_kind": readout["judge_kind"], "judge_model": readout["judge_model"],
        "arms_used": list(S1B_ARMS), "excluded_items": sorted(args.exclude_item),
        "response_cap": cap_record,
        "config": {"nu": args.nu, "mu": args.mu, "ridge": args.ridge,
                   "contemporaneous_v": args.contemporaneous_v,
                   "n_folds": args.n_folds, "folds_to_pass": args.folds_to_pass,
                   "late_from": args.late_from, "seed": args.seed,
                   "n_bootstrap": N_BOOTSTRAP},
        "n_trajectories": len({r["trajectory_id"] for r in rows}),
        "n_transitions": int(dataset["Z"].shape[0]),
        "operator": {"A": model.A.tolist(), "B": model.B.tolist(), "b": model.b.tolist(),
                     "C": model.C.tolist()},
        "fold_evaluation": folds,
        "state_provenance": state_provenance(rows),
        "g_s2_1": gate_s2_1(folds, args.folds_to_pass, args.n_folds),
        "g_s2_2": gate_s2_2(b),
        "g_s2_3": gate_s2_3(diagnostics, model.state_dim),
        "g_s2_4": gate_s2_4(b["B"], sd, {
            "one_reminder_B": abs(b["B"]),
            "s1a_late_window_contrast": s1a_late,
            "pilot_target_0.10": 0.10,
        }),
        "gold_coverage": arm_report["gold_coverage"],
        "caveat": (
            f"{arm_report['gold_coverage']['share_outside_calibration_set']:.0%} of the judged "
            "constraints lie outside the judge's calibration set. That observation noise biases "
            "the state coefficient toward zero -- the direction that would fake this line's "
            "death -- so a weak G-S2-1 is reported and handed back, never used to close "
            "(screening section 10 item 9)."),
    }

    print(f"{report['n_trajectories']} trajectories, {report['n_transitions']} transitions "
          f"({', '.join(S1B_ARMS)}), contemporaneous_v={args.contemporaneous_v}")
    print(f"A={model.A.tolist()}  B={model.B.tolist()}  b={model.b.tolist()}\n")
    for name in ("const", "turn_mean", "stateless"):
        print(f"  null {name:<10} mean one-step MSE {folds[f'null_{name}_mean_y_one_step_mse']:.6f}")
    for name in ("arx", "richer_abs_sign"):
        print(f"  {name:<15} mean {folds[f'{name}_mean_y_one_step_mse']:.6f}  beats best null in "
              f"{folds[f'{name}_n_folds_beating_best_null']}/{folds['n_folds']} folds")
    g1, g2, g3, g4 = (report["g_s2_1"], report["g_s2_2"], report["g_s2_3"], report["g_s2_4"])
    print(f"\nG-S2-1 {g1['verdict']}: arx wins {g1['arx_folds_beating_best_null']}/{g1['n_folds']} "
          f"(needs {args.folds_to_pass})")
    print(f"G-S2-2 {g2['verdict']}: B {g2['B']:+.5f} CI [{g2['ci'][0]:+.5f}, {g2['ci'][1]:+.5f}], "
          f"MDE {g2['mde_at_80pct']:.5f}, effect/MDE {g2['effect_over_mde']:.2f}")
    print(f"G-S2-3 record: spectral radius {g3['spectral_radius']:.4f} (in band {g3['radius_in_band']}), "
          f"rank {g3['controllability_rank']}/{g3['state_dim']}, "
          f"gramian cond {g3['gramian_condition']:.3e}")
    print(f"G-S2-4 {g4['verdict']}: late paired sd {sd:.4f} -> trajectories/arm "
          + "  ".join(f"{k}={v}" for k, v in g4["trajectories_per_arm"].items()))
    prov = report["state_provenance"]
    print(f"\nstate provenance (diagnostic, not a gate): y_prev coefficient pooled "
          f"{prov['pooled']['y_prev_coefficient']:+.4f} -> item-demeaned "
          f"{prov['item_demeaned']['y_prev_coefficient']:+.4f} -> trajectory-demeaned "
          f"{prov['trajectory_demeaned']['y_prev_coefficient']:+.4f} (lower bound); "
          f"u coefficient {prov['pooled']['u_coefficient']:+.4f} / "
          f"{prov['item_demeaned']['u_coefficient']:+.4f} / "
          f"{prov['trajectory_demeaned']['u_coefficient']:+.4f}")
    print(f"\n{report['caveat']}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
