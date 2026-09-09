#!/usr/bin/env python3
"""Lifted (and bilinear) Koopman-with-control on the EK-A counterfactual-branch
arm -- the properly-posed version of what scripts/fit_koopman_ergo_branch.py
fits as a scalar ARX.

Why this exists (docs/experiments/ergo_multiturn_reliability_pilot.md, section
"G-EKA-2 的失效定位"): the scalar model `c_next = const + a*c_prev + B*u + g*sf`
cannot express a state-dependent input gain, and a constant `B` makes the
closed loop provably degenerate -- under a terminal objective the optimal k
resets are the last k turns for EVERY trajectory, so an MPC on that operator
is a fixed schedule with extra steps. Feedback can only pay off if the gain
depends on where the conversation is, which in operator form is the bilinear
term N: psi_next = A psi + B u + N psi u + c.

The counterfactual-branch arm is the right data for exactly this: both actions
are generated from a byte-identical prefix, so every lifted state contributes
one u=0 and one u=1 transition and N is identified from a balanced design
rather than from whatever states an observational arm happened to visit.

The three nulls are copied verbatim from fit_koopman_ergo_branch.py (which
took them from fit_koopman_ergo_closeness.py) so the skill numbers stay
comparable to G-EKA-3's, RC-1's and K2's. Scored on the `closeness`
coordinate's one-step error, same as those.

CPU only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.ergo_math_judge import extract_answer_by_regex  # noqa: E402

N_FOLDS = 20
FOLD_SEED = 0
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 0

# Lift coordinates, in order. Every one is computable at decision time for turn
# k from turns <= k-1, and every one is a plain text statistic already on the
# row -- no new judge, no new model call. `closeness` stays coordinate 0 so the
# scored quantity is literally the same one G-EKA-3 scored.
LIFT_NAMES = ("closeness", "placeholder", "no_number", "echo_jaccard",
              "echo_exact", "log_len", "refusal", "shard_frac")
SHARD_FRAC_IDX = LIFT_NAMES.index("shard_frac")


def lift_names(with_entropy: bool) -> tuple[str, ...]:
    """`entropy_mean` is appended AFTER shard_frac so SHARD_FRAC_IDX -- and the
    truth-override the rollout does with it -- stay correct either way."""
    return (*LIFT_NAMES, "entropy_mean") if with_entropy else LIFT_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branch-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-path", type=pathlib.Path, required=True, help="JSON report; must not already exist.")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--entropy-path", type=pathlib.Path, default=None,
                   help="analyze_ergo_entropy_readout.py report for the SAME base trajectories. "
                        "When given, `entropy_mean` joins the lift as a coordinate and premise (iii) "
                        "is additionally asked directly on the entropy axis -- the state variable the "
                        "upstream literature actually triggers on (ERGO resets on entropy spikes, not "
                        "on task progress). `entropy_answer_span` is reported but kept OUT of the lift: "
                        "its IQR is 0 (the placeholder answer pins it), the same degeneracy that sank "
                        "`closeness`.")
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3],
                   help="reset budgets k the policy-separability check enumerates over")
    return p.parse_args()


def load_entropy(path: pathlib.Path) -> dict[tuple[str, int], dict]:
    report = json.loads(path.read_text())
    if report.get("prompt_profile") != "upstream":
        raise SystemExit(f"{path} was computed under prompt_profile="
                         f"{report.get('prompt_profile')!r}; the branch arm is 'upstream'")
    return {(r["trajectory_id"], r["turn"]): r for r in report["rows"]}


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def lift(row: dict, prev_row: dict | None, entropy: float | None = None) -> np.ndarray:
    """psi(row): the lifted observable vector for the state AFTER `row`'s turn.
    `prev_row` is the turn before it on the same branch, needed for the echo
    coordinates (absent on turn 1 -> echo is 0)."""

    extracted = extract_answer_by_regex(row["agent_message"])
    try:
        value = None if extracted is None else float(str(extracted).replace(",", ""))
    except (TypeError, ValueError):
        value = None
    prev_msg = "" if prev_row is None else prev_row["agent_message"]
    return np.array([
        float(row["closeness"]),
        float(value == 0),
        float(value is None),
        _jaccard(row["agent_message"], prev_msg) if prev_row is not None else 0.0,
        float(prev_row is not None and row["agent_message"] == prev_msg),
        np.log1p(len(row["agent_message"].split())) / 10.0,
        float(row["refusal_flag"]),
        row["turn"] / row["num_shards"],
        *(() if entropy is None else (entropy,)),
    ])


def build_transitions(branch_dir: pathlib.Path, entropy: dict | None = None) -> list[dict]:
    """One row per (turn, action), same construction as
    fit_koopman_ergo_branch.build_transitions -- but carrying the lifted state
    before and after. Turn 1 is dropped (no prior state); so is turn 2, whose
    prior lifted state would have no echo coordinate."""

    base = [json.loads(l) for l in (branch_dir / "trajectories.jsonl").open() if l.strip()]
    cf = [json.loads(l) for l in (branch_dir / "counterfactual_pairs.jsonl").open() if l.strip()]
    base_by = {(r["trajectory_id"], r["turn"]): r for r in base}

    out = []
    for c in cf:
        tid, k = c["base_trajectory_id"], c["branch_turn"]
        b, prev, prev2 = (base_by.get((tid, k)), base_by.get((tid, k - 1)), base_by.get((tid, k - 2)))
        if b is None:
            raise SystemExit(f"counterfactual {c['trajectory_id']} has no base row")
        if prev is None or prev2 is None:
            continue
        e_prev = None
        if entropy is not None:
            rec = entropy.get((tid, k - 1))
            if rec is None or rec["entropy_mean"] is None:
                continue
            e_prev = float(rec["entropy_mean"])
        psi_prev = lift(prev, prev2, e_prev)
        for row in (b, c):
            e_next = None
            if entropy is not None:
                rec = entropy.get((tid, k))
                e_next = e_prev if rec is None or rec["entropy_mean"] is None else float(rec["entropy_mean"])
            out.append({
                "item_id": b["item_id"], "seed": b["seed"], "trajectory_id": tid,
                "turn": k, "num_shards": b["num_shards"], "shard_frac": k / b["num_shards"],
                "psi_prev": psi_prev, "u": float(row["u_reset"]),
                "psi_next": lift(row, prev, e_next), "c_next": float(row["closeness"]),
            })
    return out


def design(rows: list[dict], model: str) -> np.ndarray:
    psi = np.array([r["psi_prev"] for r in rows])
    u = np.array([r["u"] for r in rows])[:, None]
    ones = np.ones((len(rows), 1))
    if model == "arx_scalar":  # the EK-A operator, for continuity
        return np.c_[ones, psi[:, 0], u, psi[:, SHARD_FRAC_IDX]]
    if model == "lift_linear":
        return np.c_[ones, psi, u]
    if model == "lift_bilinear":
        return np.c_[ones, psi, u, psi * u]
    raise ValueError(model)


def ridge_beta(X: np.ndarray, Y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ Y)


def null_predictions(train: list[dict], held: list[dict]) -> dict[str, np.ndarray]:
    """Copied from fit_koopman_ergo_branch._null_mses, returning predictions
    instead of MSEs so the per-row errors can be bootstrapped by item."""
    train_y = np.array([r["c_next"] for r in train])
    global_mean = float(train_y.mean())
    turn_mean = {t: float(np.mean([r["c_next"] for r in train if r["turn"] == t]))
                 for t in sorted({r["turn"] for r in train})}
    Xtr = np.array([[1.0, r["shard_frac"], r["u"]] for r in train])
    beta = np.linalg.lstsq(Xtr, train_y, rcond=None)[0]
    Xh = np.array([[1.0, r["shard_frac"], r["u"]] for r in held])
    return {
        "constant": np.full(len(held), global_mean),
        "turn_mean": np.array([turn_mean.get(r["turn"], global_mean) for r in held]),
        "stateless_ols": Xh @ beta,
    }


def cross_val_errors(rows: list[dict], ridge: float) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    """Item-disjoint 20-fold. Returns per-row squared errors for every model
    (aligned to one row order), the item label per row, and the per-fold MSEs.

    `best_null` is the winning null CHOSEN PER FOLD by that fold's MSE -- the
    same convention fit_koopman_ergo_branch.py's skill uses. Taking a per-row
    minimum instead would build an oracle that switches nulls row by row and
    that no null model can actually be."""
    items = sorted({r["item_id"] for r in rows})
    rng = np.random.default_rng(FOLD_SEED)
    shuffled = list(items)
    rng.shuffle(shuffled)
    folds = np.array_split(np.array(shuffled, dtype=object), N_FOLDS)

    names = ["constant", "turn_mean", "stateless_ols", "arx_scalar", "lift_linear", "lift_bilinear"]
    err: dict[str, list[float]] = {n: [] for n in [*names, "best_null"]}
    groups, per_fold = [], []
    for f in folds:
        held_items = set(f.tolist())
        train = [r for r in rows if r["item_id"] not in held_items]
        held = [r for r in rows if r["item_id"] in held_items]
        if not held or not train:
            continue
        yh = np.array([r["c_next"] for r in held])
        preds = null_predictions(train, held)
        for m in ("arx_scalar", "lift_linear", "lift_bilinear"):
            beta = ridge_beta(design(train, m), np.array([r["c_next"] for r in train]), ridge)
            preds[m] = design(held, m) @ beta
        fold_mse = {n: float(np.mean((yh - preds[n]) ** 2)) for n in names}
        winner = min(("constant", "turn_mean", "stateless_ols"), key=lambda n: fold_mse[n])
        for n in names:
            err[n].extend(((yh - preds[n]) ** 2).tolist())
        err["best_null"].extend(((yh - preds[winner]) ** 2).tolist())
        groups.extend([r["item_id"] for r in held])
        per_fold.append({"n_held": len(held), "best_null": winner, **fold_mse})
    return {n: np.array(v) for n, v in err.items()}, np.array(groups), per_fold


def item_boot_ci(diff: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(N_BOOTSTRAP):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        draws.append(float(diff[sel].mean()))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def fit_operator(rows: list[dict], model: str, ridge: float) -> np.ndarray:
    """Full lifted operator: predicts the WHOLE psi_next, which is what a
    rollout needs (the scalar fit only ever predicted coordinate 0)."""
    return ridge_beta(design(rows, model), np.array([r["psi_next"] for r in rows]), ridge)


def rollout_terminal(psi0: np.ndarray, schedule: list[int], turn0: int, num_shards: int,
                     beta: np.ndarray, model: str) -> float:
    """Predicted closeness at the end of the episode under `schedule`.
    shard_frac is truth-overridden at every step rather than let free-run --
    it is deterministic and known at decision time (the same override
    ergo_koopman_mpc._simulate does, signal_resolution_plan.md section 4.1)."""
    psi = psi0.copy()
    for offset, u in enumerate(schedule):
        turn = turn0 + offset
        row = [{"psi_prev": psi, "u": float(u), "shard_frac": turn / num_shards}]
        psi = (design(row, model) @ beta)[0]
        psi[SHARD_FRAC_IDX] = turn / num_shards
    return float(psi[0])


def best_schedule(psi0: np.ndarray, turn0: int, num_shards: int, horizon: int, budget: int,
                  beta: np.ndarray, model: str) -> tuple[int, ...]:
    best, best_v = None, -np.inf
    for turns in itertools.combinations(range(horizon), budget):
        sched = [1 if i in turns else 0 for i in range(horizon)]
        v = rollout_terminal(psi0, sched, turn0, num_shards, beta, model)
        if v > best_v:
            best, best_v = turns, v
    return best


def policy_separability(rows: list[dict], beta: np.ndarray, model: str, budgets: list[int]) -> dict:
    """THE question EK-B hinges on: does the identified operator produce a
    schedule that depends on the trajectory, or the same turns for everyone?
    If the latter, closed loop IS open loop here and the MPC arm cannot differ
    from a fixed schedule no matter how many GPU-hours it gets.

    Enumerated exactly (T <= 12, k <= 3) rather than through the receding-horizon
    MPC, which is an approximation of this same argmax."""
    starts = {}
    for r in rows:  # earliest transition of each trajectory = its decision point
        key = r["trajectory_id"]
        if key not in starts or r["turn"] < starts[key]["turn"]:
            starts[key] = r
    out = {}
    for k in budgets:
        chosen = {}
        for key, r in starts.items():
            horizon = r["num_shards"] - r["turn"] + 1
            if horizon < k:
                continue
            sched = best_schedule(r["psi_prev"], r["turn"], r["num_shards"], horizon, k, beta, model)
            # express as offsets from the END, so trajectories of different
            # length are comparable ("reset on the last turn" is one schedule)
            chosen[key] = tuple(sorted(horizon - 1 - t for t in sched))
        counts: dict[tuple, int] = {}
        for v in chosen.values():
            counts[v] = counts.get(v, 0) + 1
        modal, modal_n = max(counts.items(), key=lambda kv: kv[1]) if counts else ((), 0)
        out[k] = {
            "n_trajectories": len(chosen),
            "n_distinct_schedules": len(counts),
            "modal_schedule_from_end": list(modal),
            "modal_share": modal_n / len(chosen) if chosen else None,
            "schedule_counts": {str(list(kk)): v for kk, v in sorted(counts.items(), key=lambda kv: -kv[1])},
        }
    return out


def entropy_axis_premise_iii(rows: list[dict], entropy: dict, branch_dir: pathlib.Path) -> dict:
    """G-EKA-2 re-asked on the state axis upstream actually uses. `delta_c` is
    the same measured one-step causal gain; only the regressor changes.

    Reports the design's MDE alongside every slope, because a CI crossing 0
    without one proves nothing (failure mode 29) -- and that is exactly how
    the closeness version of this test failed."""

    base = [json.loads(l) for l in (branch_dir / "trajectories.jsonl").open() if l.strip()]
    base_by = {(r["trajectory_id"], r["turn"]): r for r in base}
    cf = [json.loads(l) for l in (branch_dir / "counterfactual_pairs.jsonl").open() if l.strip()]

    pairs = []
    for c in cf:
        tid, k = c["base_trajectory_id"], c["branch_turn"]
        b, prev = base_by.get((tid, k)), base_by.get((tid, k - 1))
        if b is None or prev is None:
            continue
        rec = entropy.get((tid, k - 1))
        if rec is None:
            continue
        reset_row, plain_row = (c, b) if c["u_reset"] else (b, c)
        pairs.append({"item_id": b["item_id"],
                      "delta_c": reset_row["closeness"] - plain_row["closeness"],
                      "shard_frac": k / b["num_shards"], "c_prev": prev["closeness"],
                      "entropy_mean": rec["entropy_mean"],
                      "entropy_answer_span": rec["entropy_answer_span"]})

    d = np.array([p["delta_c"] for p in pairs])
    g = np.array([p["item_id"] for p in pairs])
    out = {"n_pairs": len(pairs), "n_items": int(len(np.unique(g)))}
    for col in ("entropy_mean", "entropy_answer_span"):
        usable = [p for p in pairs if p[col] is not None]
        dd = np.array([p["delta_c"] for p in usable])
        x = np.array([p[col] for p in usable])
        gg = np.array([p["item_id"] for p in usable])
        X = np.c_[np.ones(len(x)), x]
        beta = np.linalg.lstsq(X, dd, rcond=None)[0]
        lo, hi = item_boot_ci_stat(gg, lambda sel: float(
            np.linalg.lstsq(np.c_[np.ones(len(sel)), x[sel]], dd[sel], rcond=None)[0][1]))
        se = (hi - lo) / (2 * 1.96)
        out[col] = {"n": len(usable), "sd_of_regressor": float(x.std()),
                    "iqr_of_regressor": float(np.percentile(x, 75) - np.percentile(x, 25)),
                    "corr_with_shard_frac": float(np.corrcoef(x, [p["shard_frac"] for p in usable])[0, 1]),
                    "intercept": float(beta[0]), "slope": float(beta[1]), "ci_slope": [lo, hi],
                    "mde_80": float(2.8 * se), "pass": bool(lo * hi > 0)}

        # THE control that decides whether this is a state axis at all: the
        # nulls and the operator already receive `shard_frac`, a deterministic
        # exogenous quantity known before the conversation starts. A regressor
        # that only proxies it is not feedback-usable state, however tight its
        # marginal CI is.
        for name, extra in (("controlled_for_shard_frac", [np.array([p["shard_frac"] for p in usable])]),
                            ("controlled_for_shard_frac_and_c_prev",
                             [np.array([p["shard_frac"] for p in usable]),
                              np.array([p["c_prev"] for p in usable])])):
            cols = [x, *extra]
            M = np.column_stack([np.ones(len(dd))] + cols)
            b_ctrl = np.linalg.lstsq(M, dd, rcond=None)[0]
            clo, chi = item_boot_ci_stat(gg, lambda sel, cols=cols: float(np.linalg.lstsq(
                np.column_stack([np.ones(len(sel))] + [c[sel] for c in cols]), dd[sel], rcond=None)[0][1]))
            out[col][name] = {"slope": float(b_ctrl[1]), "ci_slope": [clo, chi],
                              "mde_80": float(2.8 * (chi - clo) / (2 * 1.96)),
                              "attenuation_vs_marginal": float(1 - abs(b_ctrl[1]) / abs(beta[1])) if beta[1] else None,
                              "pass": bool(clo * chi > 0)}

    # The exogenous ramp on its own, for the comparison that matters: it is
    # what the entropy slope turns out to be a proxy for.
    sfv = np.array([p["shard_frac"] for p in pairs])
    M = np.c_[np.ones(len(d)), sfv]
    b_sf = np.linalg.lstsq(M, d, rcond=None)[0]
    slo, shi = item_boot_ci_stat(g, lambda sel: float(
        np.linalg.lstsq(np.c_[np.ones(len(sel)), sfv[sel]], d[sel], rcond=None)[0][1]))
    out["shard_frac_alone"] = {"slope": float(b_sf[1]), "ci_slope": [slo, shi],
                               "mde_80": float(2.8 * (shi - slo) / (2 * 1.96)),
                               "pass": bool(slo * shi > 0)}
    out["mean_delta_c"] = float(d.mean())
    return out


def item_boot_ci_stat(groups: np.ndarray, stat) -> tuple[float, float]:
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(N_BOOTSTRAP):
        sel = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        try:
            draws.append(stat(sel))
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")
    entropy = load_entropy(args.entropy_path) if args.entropy_path else None
    rows = build_transitions(args.branch_dir, entropy)
    names = lift_names(entropy is not None)
    items = sorted({r["item_id"] for r in rows})
    print(f"{len(rows)} transitions from {len(items)} items ({len(rows)//2} same-prefix pairs; "
          f"turns 1-2 dropped: no prior lifted state)")
    print(f"lift ({len(names)}d): {', '.join(names)}")
    psi = np.array([r["psi_prev"] for r in rows])
    print("\n=== lift coordinate variation (the thing closeness alone did not have) ===")
    for j, nm in enumerate(names):
        col = psi[:, j]
        print(f"  {nm:14s} mean={col.mean():+.4f} sd={col.std():.4f} "
              f"distinct={len(np.unique(col.round(6))):4d} IQR={np.percentile(col,75)-np.percentile(col,25):.4f}")

    if entropy is not None:
        prem = entropy_axis_premise_iii(rows, entropy, args.branch_dir)
        print(f"\n=== premise (iii) on the ENTROPY axis (delta_c ~ entropy_{{k-1}}), "
              f"n={prem['n_pairs']} pairs / {prem['n_items']} items ===")
        for col in ("entropy_mean", "entropy_answer_span"):
            c = prem[col]
            print(f"  {col:22s} n={c['n']:4d} sd(x)={c['sd_of_regressor']:.4f} IQR(x)={c['iqr_of_regressor']:.4f}")
            print(f"    marginal              slope={c['slope']:+.4f}  CI=[{c['ci_slope'][0]:+.4f}, {c['ci_slope'][1]:+.4f}]  "
                  f"MDE@80%=±{c['mde_80']:.4f}  -> {'PASS' if c['pass'] else 'FAIL'}   "
                  f"(corr with shard_frac = {c['corr_with_shard_frac']:+.4f})")
            for name in ("controlled_for_shard_frac", "controlled_for_shard_frac_and_c_prev"):
                cc = c[name]
                print(f"    {name:22s}slope={cc['slope']:+.4f}  CI=[{cc['ci_slope'][0]:+.4f}, {cc['ci_slope'][1]:+.4f}]  "
                      f"MDE@80%=±{cc['mde_80']:.4f}  -> {'PASS' if cc['pass'] else 'FAIL'}   "
                      f"(attenuated {cc['attenuation_vs_marginal']*100:.0f}% vs marginal)")
        sfa = prem["shard_frac_alone"]
        print(f"  shard_frac alone (the deterministic ramp every null already gets)")
        print(f"    slope={sfa['slope']:+.4f}  CI=[{sfa['ci_slope'][0]:+.4f}, {sfa['ci_slope'][1]:+.4f}]  "
              f"MDE@80%=±{sfa['mde_80']:.4f}  -> {'PASS' if sfa['pass'] else 'FAIL'}")

    err, groups, per_fold = cross_val_errors(rows, args.ridge)
    print(f"\n=== held-out one-step MSE on `closeness` ({len(per_fold)} item-disjoint folds) ===")
    for n in ("constant", "turn_mean", "stateless_ols", "arx_scalar", "lift_linear", "lift_bilinear"):
        print(f"  {n:16s} {err[n].mean():.6f}")
    best_null = err["best_null"]
    print(f"  best null chosen per fold: " +
          ", ".join(f"{n}x{sum(1 for f in per_fold if f['best_null'] == n)}"
                    for n in ("constant", "turn_mean", "stateless_ols")
                    if any(f["best_null"] == n for f in per_fold)))

    tests = {}
    for label, a, b in (
        ("T0  arx_scalar vs best null    (EK-A's own model, on this sample)", best_null, err["arx_scalar"]),
        ("T1  lift_linear vs best null   (is there dynamics in the lift?)", best_null, err["lift_linear"]),
        ("T2  lift_bilinear vs lift_linear (STATE-DEPENDENT GAIN: premise (iii))", err["lift_linear"], err["lift_bilinear"]),
        ("T3  lift_linear vs arx_scalar  (did lifting buy anything?)", err["arx_scalar"], err["lift_linear"]),
    ):
        diff = a - b  # >0 means the second model has lower error
        lo, hi = item_boot_ci(diff, groups)
        tests[label] = {"mean_skill": float(diff.mean()), "ci": [lo, hi], "pass": bool(lo * hi > 0)}
        print(f"\n{label}\n  skill={diff.mean():+.6f}  CI=[{lo:+.6f}, {hi:+.6f}]  -> {'CI excludes 0' if lo*hi>0 else 'CI crosses 0'}")

    print("\n=== policy separability: does the operator want a state-dependent schedule? ===")
    sep = {}
    for model in ("lift_linear", "lift_bilinear"):
        beta = fit_operator(rows, model, args.ridge)
        sep[model] = policy_separability(rows, beta, model, args.budgets)
        for k, s in sep[model].items():
            print(f"  {model:14s} budget k={k}: {s['n_distinct_schedules']} distinct schedules over "
                  f"{s['n_trajectories']} trajectories; modal {s['modal_schedule_from_end']} "
                  f"(offsets from the last turn) holds {s['modal_share']*100:.1f}%")

    report = {
        "branch_dir": str(args.branch_dir), "n_transitions": len(rows), "n_items": len(items),
        "lift_names": list(names), "ridge": args.ridge,
        "entropy_path": str(args.entropy_path) if args.entropy_path else None,
        "entropy_axis_premise_iii": prem if entropy is not None else None,
        "mean_mse": {n: float(v.mean()) for n, v in err.items()},
        "tests": tests, "policy_separability": sep, "per_fold": per_fold,
        "n_folds": len(per_fold), "fold_seed": FOLD_SEED,
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {args.out_path}")


if __name__ == "__main__":
    main()
