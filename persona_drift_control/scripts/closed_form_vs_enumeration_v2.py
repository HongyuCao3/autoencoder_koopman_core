#!/usr/bin/env python3
"""#3 v2: per-decision re-solve over the four discrete actions, no relaxation.

Spec: docs/experiments/controllability_closed_form_evidence3.md section 5. Same
arm, operator, beta, lambda, H and 2297 decisions as v1
(`closed_form_vs_enumeration.py`, whose replay/enumeration code is reused
unchanged). At every decision the first action is the exact argmin of J over
{copy, step_down, half_step_down, paraphrase} -- no continuous one-hot
relaxation, no projection (v1 gap 2). Ties break in candidate order, as the
deployed enumeration does.

How the H-1 later steps are filled (section 5.3; ruling 2026-11-05, made after
the numbers were seen):
  A (GATE)        later steps repeat the first command c;
  B (exploratory) later steps are `copy` (one step, then stop) -- post hoc,
                  never enters the verdict.

Before anything is compared the replay must reproduce the logged action at
every decision and the decision count must equal v1's summary.json; otherwise
the script raises.

Zero GPU, read-only on outputs/; writes
outputs/closed_form_check_tsar_cefr_v2/summary.json and refuses to overwrite.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_cf_v1", SCRIPTS / "closed_form_vs_enumeration.py")
v1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v1)

from persona_drift.run_provenance import provenance  # noqa: E402  (path set by v1)

V1_SUMMARY = v1.OUT_DIR / "summary.json"
OUT_DIR = v1.PDC_ROOT / "outputs" / "closed_form_check_tsar_cefr_v2"
FILLS = {
    "A_hold_first_command": lambda a, h: (a,) * h,
    "B_first_then_copy": lambda a, h: (a,) + ("copy",) * (h - 1),
}
GATE_FILL = "A_hold_first_command"


def discrete_first(costs, names, fill, horizon=v1.HORIZON) -> str:
    """argmin over the candidate first actions of the cost of fill(a); ties -> candidate order."""
    vals = [costs[fill(a, horizon)] for a in names]
    return names[int(np.argmin(vals))]


def run(arm_dir=v1.ARM_DIR, operator=v1.OPERATOR, v1_summary=V1_SUMMARY) -> dict:
    cfg = json.loads((arm_dir / "run_config.json").read_text())
    beta, lam = float(cfg["rulings"]["beta"]), float(cfg["rulings"]["lambda_cost"])
    fit = json.loads(operator.read_text())["final_operator"]
    K, B, c = (np.asarray(fit[k], dtype=float) for k in ("K", "B", "c"))
    cands = v1.candidates(fit["u_layout"])
    names = list(cands)
    rows = [json.loads(line) for line in (arm_dir / "trajectories.jsonl").open()]

    n = replay_ok = 0
    per = {f: {"agree": 0, "gaps": [], "mismatches": {}} for f in FILLS}
    for xi, target, logged in v1.decisions(rows, int(cfg["plan"]["max_steps"])):
        star = float(v1.CEFR_CODE[target.upper()])
        costs = v1.sequence_costs(K, B, c, xi, star, beta, lam, cands)
        enum = v1.enumerate_first(costs)
        n += 1
        replay_ok += enum == logged
        best = min(costs.values())
        for f, fill in FILLS.items():
            d = discrete_first(costs, names, fill)
            if d == enum:
                per[f]["agree"] += 1
            else:
                per[f]["gaps"].append(min(v for seq, v in costs.items() if seq[0] == d) - best)
                key = f"{enum}->{d}"
                per[f]["mismatches"][key] = per[f]["mismatches"].get(key, 0) + 1

    ref = json.loads(v1_summary.read_text())
    if replay_ok != n or n != ref["n_decisions"]:
        raise RuntimeError(f"replay {replay_ok}/{n} vs v1 n={ref['n_decisions']}: comparison meaningless")

    fills = {f: {"agreement_rate": p["agree"] / n,
                 "median_delta_J_when_disagree": float(np.median(p["gaps"])) if p["gaps"] else None,
                 "mismatch_counts_enum_to_discrete": p["mismatches"]} for f, p in per.items()}
    rate = fills[GATE_FILL]["agreement_rate"]
    return {
        "probe": "closed_form_vs_enumeration_v2", "line": "tsar_cefr",
        "provenance": provenance(switches={"arm_dir": str(arm_dir), "operator": str(operator),
                                           "horizon": v1.HORIZON, "python": sys.executable}),
        "beta": beta, "lambda_cost": lam, "u_layout": fit["u_layout"],
        "n_decisions": n, "replay_reproduces_logged_action": replay_ok / n,
        "v1_agreement_rate": ref["agreement_rate"],
        "gate_fill": GATE_FILL, "fills": fills,
        "verdict": "prelim" if rate >= v1.AGREEMENT_PASS else "future_work",
        "rule": (f"fill A agreement >= {v1.AGREEMENT_PASS} -> Thrust 3 prelim; else future work "
                 "(spec 5.2). Fill B is exploratory and chosen post hoc (spec 5.3)."),
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
