#!/usr/bin/env python3
"""#4a: is the command column B a measurement on the behavioural lines?

Spec: docs/experiments/command_channel_identifiability_plan.md section 2. The
rank check (`_identifiability`, copied from the M1 probes) says whether B is
DEFINED; this adds whether it is PINNED: refit the Table 1 operator on 2000
trajectory-level bootstrap resamples and report, per element of B, the 95%
interval and how often the sign agrees with the full-data fit.

"Usable for a Gramian" (spec 2.3) = rank check passes AND every main element's
interval excludes 0 AND its sign stability >= 0.95. The main elements are the
entries of C @ B -- the one-step effect of each command column on the readout
y. On `constraint` / `defense` C is a selection vector, so C @ B is the row of
B that lands on the newest y; on `tsar_cefr` it is the per-action effect on
the CEFR level. The rest of B (carry into older taps, into s) is reported but
does not enter the verdict.

The resampling unit is the trajectory, as the spec signs. Trajectories that
share an item are not independent, so the item-level interval (Table 1's
unit) is written beside it; if the two verdicts disagree, the artifact says so
rather than picking one.

Zero GPU, read-only on `outputs/`; writes `results/b_identifiability/<line>.json`
at the repo root and refuses to overwrite.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

PDC_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PDC_ROOT.parent
sys.path.insert(0, str(PDC_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from persona_drift.modeling.dataset import ReducedStateConfig  # noqa: E402
from persona_drift.modeling.koopman import KoopmanSurrogate, no_extra_features  # noqa: E402
from persona_drift.run_provenance import provenance  # noqa: E402


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, PDC_ROOT / "scripts" / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pse = _load("_paper_side_experiments_behavioral", "paper_side_experiments_behavioral.py")
pst = _load("_paper_side_experiments_tsar_cefr", "paper_side_experiments_tsar_cefr.py")
tsar = pst.tsar

N_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
SIGN_STABILITY_MIN = 0.95
LINES = ("constraint", "defense", "tsar_cefr")
OUT_DIR = REPO_ROOT / "results" / "b_identifiability"


def _behavioral_units(name: str):
    """[(trajectory_id, item, transitions)] on Table 1's state config."""
    spec, _, by_traj, _, nu, horizon, refused = pse._setup(name)
    cfg = ReducedStateConfig(nu=nu, mu=1, contemporaneous_v=True)
    units = []
    for tid, rows in by_traj.items():
        data = pse._dataset({tid: rows}, cfg, spec["y_col"], spec["u_col"])
        units.append((str(tid), str(rows[0][spec["item_col"]]), data))
    meta = {"judge": spec["judge"], "nu": nu, "mu": 1, "contemporaneous_v": True,
            "horizon": horizon, "n_trajectories_refused_unparsed_readout": refused,
            "command_columns": [spec["u_col"]]}
    return units, meta


def _tsar_units():
    trajectories = tsar.build_trajectories(tsar.read_jsonl(tsar.DEFAULT_ROWS_PATH))
    refused = sorted(k for k, t in trajectories.items() if t["has_nan"])
    units = []
    for key in sorted(trajectories):
        traj = trajectories[key]
        if traj["has_nan"]:
            continue
        data = tsar.to_dataset(tsar.transitions(traj, tsar.LAG))
        pst._assert_action_dofs(data["V"])
        units.append((f"{key[0]}|{key[1]}", str(traj["source_id"]), data))
    meta = {"judge": "fixed CEFR classifier (not the controlled model's self-judgement)",
            "lag": tsar.LAG, "horizon": tsar.HORIZON,
            "n_trajectories_refused_unparsed_readout": len(refused),
            "command_columns": [f"u_{a}" for a in tsar.ACTIONS],
            "reference_action": tsar.REFERENCE_ACTION}
    return units, meta


def _stack(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in ("Z", "V", "Z_next", "Y")}


def _fit(data: dict):
    model = KoopmanSurrogate(extra_features_fn=no_extra_features).fit(data)
    B = np.asarray(model.B)
    return B, (np.asarray(model.C) @ B).reshape(-1)


def _intervals(draws: np.ndarray, point: np.ndarray) -> list[dict]:
    flat_draws = draws.reshape(draws.shape[0], -1)
    flat_point = point.reshape(-1)
    out = []
    for j, p in enumerate(flat_point):
        lo, hi = np.quantile(flat_draws[:, j], [0.025, 0.975])
        out.append({
            "point": float(p), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "sign_stability": float(np.mean(np.sign(flat_draws[:, j]) == np.sign(p))),
        })
    return out


def _bootstrap(units, unit_of, *, seed: int, n_resamples: int):
    groups = sorted({unit_of(u) for u in units})
    members = {g: [u[2] for u in units if unit_of(u) == g] for g in groups}
    rng = np.random.default_rng(seed)
    B_draws, CB_draws = [], []
    for _ in range(n_resamples):
        picked = rng.choice(len(groups), size=len(groups), replace=True)
        B, CB = _fit(_stack([d for i in picked for d in members[groups[i]]]))
        B_draws.append(B)
        CB_draws.append(CB)
    return np.stack(B_draws), np.stack(CB_draws), len(groups)


def _verdict(identified: bool, main: list[dict]) -> bool:
    return bool(identified and all(m["excludes_zero"] and m["sign_stability"] >= SIGN_STABILITY_MIN
                                   for m in main))


def run_line(name: str, *, n_resamples: int = N_RESAMPLES) -> dict:
    units, meta = _tsar_units() if name == "tsar_cefr" else _behavioral_units(name)
    full = _stack([u[2] for u in units])
    B, CB = _fit(full)
    ident = pse._identifiability(full["Z"], full["V"])

    by_unit = {}
    for label, unit_of in (("trajectory", lambda u: u[0]), ("item", lambda u: u[1])):
        B_draws, CB_draws, n_groups = _bootstrap(units, unit_of, seed=BOOTSTRAP_SEED,
                                                 n_resamples=n_resamples)
        main = _intervals(CB_draws, CB)
        by_unit[label] = {
            "n_groups": n_groups,
            "main_elements_CB": main,
            "B_elements": _intervals(B_draws, B),
            "usable_for_gramian": _verdict(ident["identified"], main),
        }

    signed, robustness = by_unit["trajectory"], by_unit["item"]
    out = {
        "probe": "b_identifiability_bootstrap",
        "line": name,
        "provenance": provenance(switches={"n_resamples": n_resamples, "bootstrap_seed": BOOTSTRAP_SEED,
                                           "python": sys.executable}),
        **meta,
        "n_trajectories": len(units),
        "n_transitions": int(len(full["Z"])),
        "B_shape": list(B.shape),
        "identifiability": ident,
        "rule": ("usable_for_gramian = rank check passes AND every C@B element's 95% interval "
                 f"excludes 0 AND sign stability >= {SIGN_STABILITY_MIN}"),
        "bootstrap": by_unit,
        "usable_for_gramian": signed["usable_for_gramian"],
        "units_agree": signed["usable_for_gramian"] == robustness["usable_for_gramian"],
    }
    if name == "defense":
        out["judge_kind"] = "self"
        out["caveat"] = ("self-judged safety (global.md named exception): one-sided miss, "
                         "systematically under-reports the effect")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", nargs="*", default=list(LINES), choices=LINES)
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.lines:
        out_path = args.out_dir / f"{name}.json"
        if out_path.exists():
            raise FileExistsError(f"{out_path} exists; outputs are never overwritten")
        result = run_line(name, n_resamples=args.n_resamples)
        out_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
        main_cb = result["bootstrap"]["trajectory"]["main_elements_CB"]
        print(f"[done] {name} identified={result['identifiability']['identified']} "
              f"usable={result['usable_for_gramian']} units_agree={result['units_agree']} -> {out_path}")
        for m in main_cb:
            print(f"       C@B {m['point']:+.4f} [{m['ci_low']:+.4f},{m['ci_high']:+.4f}] "
                  f"sign={m['sign_stability']:.3f}")


if __name__ == "__main__":
    main()
