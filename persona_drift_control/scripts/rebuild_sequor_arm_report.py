#!/usr/bin/env python3
"""Rebuild an S0-0 arm's `arm_report.json` from the rows already on disk.

WHY IT EXISTS: job 15761810 generated all 1404 rows of the fidelity-harness arm
and wrote `trajectories.jsonl` and `run_config.json`, then died building the
summary (`NameError: turn_clock`). The generation -- 52 GPU-minutes -- is
intact; only the derived report is missing, and the gate script requires it
because that is where the per-item response-cap accounting lives.

Regenerating the arm on a GPU to recover a derived file would be spending an
hour to recompute something the rows already determine. This recomputes it,
and marks the artifact as rebuilt: the timing fields cannot be recovered from
rows, so they are written as null with the reason attached rather than as
plausible-looking numbers.

CPU only. It refuses to overwrite an existing report.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from persona_drift.run_provenance import provenance  # noqa: E402
from persona_drift.sequor_trajectory import (  # noqa: E402
    CAP_CRITERION,
    assert_pairs_share_prefix,
    cap_accounting,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-dir", type=pathlib.Path, required=True)
    p.add_argument("--reason", required=True,
                   help="Why the report is being rebuilt rather than produced by the run. Written "
                        "into the artifact: a derived file with no account of its own origin is "
                        "how a rebuilt number gets quoted as a measured one.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.arm_dir / "arm_report.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")
    rows = [json.loads(l) for l in (args.arm_dir / "trajectories.jsonl").open() if l.strip()]
    config = json.loads((args.arm_dir / "run_config.json").read_text())

    n_pairs = assert_pairs_share_prefix(rows)
    caps = cap_accounting(rows, CAP_CRITERION)
    report = {
        "mode": config.get("mode"), "n_rows": len(rows), "n_pairs": n_pairs,
        "prefix_check": "every pair shares a byte-identical prefix (assert_pairs_share_prefix)",
        "seeds": config.get("seeds"), "decoding_mode": config.get("decoding_mode"),
        "decoding": config.get("decoding"),
        "constraints_in_system": config.get("constraints_in_system"),
        "max_new_tokens": config.get("max_new_tokens"),
        **caps,
        "elapsed_s": None, "seconds_per_generation": None, "batches": None,
        "rebuilt_from_rows": True, "rebuild_reason": args.reason,
        "rebuild_provenance": provenance(switches={"arm_dir": str(args.arm_dir)}),
        "timing_unavailable": "elapsed/batches are properties of the run, not of the rows; the run "
                              "that produced these rows did not get to write them",
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"{len(rows)} rows, {n_pairs} pairs, seeds {report['seeds']}")
    print(f"cap {report['max_new_tokens']}: {caps['n_hit_token_cap']}/{len(rows)} truncated "
          f"({caps['token_cap_share']*100:.1f}%), median {caps['output_tokens_median']} tokens, "
          f"max {caps['output_tokens_max']}")
    for item, stats in caps["token_cap_by_item"].items():
        flag = "  <-- OVER" if stats["token_cap_share"] > CAP_CRITERION else ""
        print(f"    {item:16s} {stats['n_hit_token_cap']:3d}/{stats['n_rows']:3d} "
              f"({stats['token_cap_share']*100:5.1f}%)  median {stats['output_tokens_median']:5d}{flag}")
    print(f"\ncap criterion {'PASSED' if caps['cap_criterion_pass'] else 'FAILED'} "
          f"(per item, {CAP_CRITERION*100:.0f}%)")
    print(f"written to {out_path}  [rebuilt_from_rows=true]")


if __name__ == "__main__":
    main()
