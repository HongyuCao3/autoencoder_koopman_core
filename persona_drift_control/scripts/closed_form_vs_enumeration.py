#!/usr/bin/env python3
"""#3: does the enumerating MPC pick what the linear structure gives in closed form?

Spec: docs/experiments/controllability_closed_form_evidence3.md section 2. At
every logged `koopman_mpc` decision of the `tsar_cefr` S3 arm, rebuild the
state xi from the log, then

  (i)  enumeration: the deployed controller's own objective, all 4^H action
       sequences (`run_tsar_cefr_closed_loop_arms.mpc_action`, reimplemented
       here vectorised and checked against the logged action -- a replay that
       does not reproduce the log would make the comparison meaningless);
  (ii) closed form: the spec's model s_(t+h) = K^h s_t + sum_j K^j (B c + b)
       holds ONE command c over the horizon, so J(c) is a quadratic in the
       relaxed one-hot c in R^3 (copy = 0; the switching cost lambda * 1[c != 0]
       equals lambda * sum(c) on every candidate). c* = -Q^-1 q / 2, projected to
       the nearest candidate.

Agreement = share of decisions where (i)'s first action equals (ii)'s. Where
they differ, dJ = (best sequence cost given (ii)'s first action) - (optimum).

Decisions where the MPC chose `copy` end the trajectory under
stop_on_arrival=on_copy and emit no row; they are inferred from a trajectory
that stops before max_steps and counted as `copy`.

Zero GPU, read-only on outputs/; writes outputs/closed_form_check_tsar_cefr/
summary.json and refuses to overwrite.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

PDC_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PDC_ROOT.parent
sys.path.insert(0, str(PDC_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from koopman_ae.core import controllability_diagnostics  # noqa: E402
from persona_drift.run_provenance import provenance  # noqa: E402

ARM_DIR = PDC_ROOT / "outputs" / "tsar_cefr_s3_arm"
OPERATOR = PDC_ROOT / "outputs" / "tsar_cefr_phase3" / "operator_fit.json"
OUT_DIR = PDC_ROOT / "outputs" / "closed_form_check_tsar_cefr"
CEFR_CODE = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}   # run_tsar_cefr_closed_loop_arms.py
HORIZON = 4                                                         # MPC_HORIZON there
AGREEMENT_PASS = 0.95


def candidates(u_layout) -> dict[str, np.ndarray]:
    out = {"copy": np.zeros(len(u_layout))}
    out.update({a: np.eye(len(u_layout))[i] for i, a in enumerate(u_layout)})
    return out


def sequence_costs(K, B, c, xi, star, beta, lam, cands, horizon=HORIZON):
    """{action sequence: cost} under the deployed objective."""
    names = list(cands)
    costs = {}
    for seq in itertools.product(names, repeat=horizon):
        state, cost = np.asarray(xi, dtype=float), 0.0
        for a in seq:
            state = K @ state + B @ cands[a] + c
            cost += (state[0] - star) ** 2 + beta * (1.0 - state[1]) + lam * (a != "copy")
        costs[seq] = cost
    return costs


def enumerate_first(costs) -> str:
    # mpc_action keeps the FIRST sequence with the strictly lowest cost, in
    # ACTION_NAMES order; itertools.product walks the same order.
    best = min(costs.values())
    return next(seq for seq, v in costs.items() if v == best)[0]


def closed_form(K, B, c, xi, star, beta, lam, cands, horizon=HORIZON):
    """Quadratic J(c) for one command held over the horizon; returns (c*, nearest candidate)."""
    n_u = B.shape[1]
    Q, q = np.zeros((n_u, n_u)), lam * horizon * np.ones(n_u)
    free = np.asarray(xi, dtype=float)
    gain = np.zeros((K.shape[0], n_u))
    for _ in range(horizon):
        free = K @ free + c
        gain = K @ gain + B
        m, n = gain[0], gain[1]
        Q += np.outer(m, m)
        q += 2.0 * (free[0] - star) * m - beta * n
    c_star = -0.5 * np.linalg.solve(Q, q)
    nearest = min(cands, key=lambda a: float(np.linalg.norm(cands[a] - c_star)))
    return c_star, nearest


def decisions(rows, max_steps: int):
    """[(xi, target, logged action)] for every MPC decision, copy-stops included.

    xi before step k is [ell_(k-1), s_(k-1), ell_(k-2), s_(k-2)], with the source
    (ell_0, s_0 = 1) standing in for the missing older tap at k = 1 -- the
    runner's initial state [ell0, 1, ell0, 1]."""
    by_traj = {}
    for r in rows:
        if r["arm"] == "koopman_mpc":
            by_traj.setdefault((r["text_id"], r["seed"]), []).append(r)
    out = []
    for steps in by_traj.values():
        steps.sort(key=lambda r: r["step"])
        ell = [float(steps[0]["level_expected"])] + [float(r["level_expected"]) for r in steps[1:]]
        s = [1.0] + [float(r["meaning_to_source"]) for r in steps[1:]]
        actions = [r["action"] for r in steps[1:]]
        if len(actions) < max_steps:
            actions.append("copy")
        for k, action in enumerate(actions, start=1):
            prev = max(k - 2, 0)
            out.append(([ell[k - 1], s[k - 1], ell[prev], s[prev]], steps[0]["target_cefr"], action))
    return out


def run(arm_dir: pathlib.Path = ARM_DIR, operator: pathlib.Path = OPERATOR) -> dict:
    cfg = json.loads((arm_dir / "run_config.json").read_text())
    beta, lam = float(cfg["rulings"]["beta"]), float(cfg["rulings"]["lambda_cost"])
    fit = json.loads(operator.read_text())["final_operator"]
    K, B, c = (np.asarray(fit[k], dtype=float) for k in ("K", "B", "c"))
    cands = candidates(fit["u_layout"])
    rows = [json.loads(line) for line in (arm_dir / "trajectories.jsonl").open()]

    n = agree = replay_ok = 0
    gaps, mismatches = [], {}
    for xi, target, logged in decisions(rows, int(cfg["plan"]["max_steps"])):
        star = float(CEFR_CODE[target.upper()])
        costs = sequence_costs(K, B, c, xi, star, beta, lam, cands)
        enum = enumerate_first(costs)
        _, cf = closed_form(K, B, c, xi, star, beta, lam, cands)
        n += 1
        replay_ok += enum == logged
        if enum == cf:
            agree += 1
        else:
            best = min(costs.values())
            gaps.append(min(v for seq, v in costs.items() if seq[0] == cf) - best)
            mismatches[f"{enum}->{cf}"] = mismatches.get(f"{enum}->{cf}", 0) + 1

    ctrb = controllability_diagnostics(K, B, horizon=HORIZON)
    rate = agree / n
    return {
        "probe": "closed_form_vs_enumeration", "line": "tsar_cefr",
        "provenance": provenance(switches={"arm_dir": str(arm_dir), "operator": str(operator),
                                           "horizon": HORIZON, "python": sys.executable}),
        "beta": beta, "lambda_cost": lam, "u_layout": fit["u_layout"],
        "n_decisions": n,
        "replay_reproduces_logged_action": replay_ok / n,
        "agreement_rate": rate,
        "median_delta_J_when_disagree": float(np.median(gaps)) if gaps else None,
        "mismatch_counts_enum_to_closed_form": mismatches,
        "verdict": "prelim" if rate >= AGREEMENT_PASS else "future_work",
        "rule": f"agreement >= {AGREEMENT_PASS} -> Thrust 3 prelim; else future work (spec 2.2)",
        "controllability": {k: ctrb[k] for k in ("controllability_rank", "controllability_singular_values",
                                                 "gramian_condition", "spectral_radius")},
        "state_dim": int(K.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()
    out_path = args.out_dir / "summary.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
    result = run()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
