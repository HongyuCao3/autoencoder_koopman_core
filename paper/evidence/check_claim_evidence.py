#!/usr/bin/env python3
"""Step 4 mechanical gate: check contract.yaml's claim_ledger against numbers.yaml.

The rule this enforces was decided on 2026-09-06 (paper/evidence/superseded.md
section 1.1 and section 5 item 3):

  * a claim may NOT cite a `superseded` number -- a better estimate of that same
    quantity exists and the project has retired this one;
  * a claim MAY cite a `caveated` number, but only while stating the caveat, so
    every claim that cites one must carry a non-empty `notes` field.

What this script cannot check: whether the claim's own prose actually states the
caveat. That is a judgment call and stays with the author. What it can do is make
the caveat impossible to not-read -- it prints the caveat text beside the claim it
is attached to, and refuses to pass while the notes field is empty.

Usage:
    python paper/evidence/check_claim_evidence.py [--contract paper/contract.yaml]
                                                  [--numbers paper/evidence/numbers.yaml]
Exit codes: 0 all gates pass, 1 a gate failed, 2 could not run (missing file).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

ID_PATTERN = re.compile(r"\bn_[a-z]+_\d+\b")


def collect_ids(value) -> list[str]:
    """Pull n_xxx ids out of whatever shape the evidence field happens to take.

    Step 4 has not fixed the evidence string convention yet (the skill template
    ships `table_cell:...` strings; the execution plan says evidence points at
    numbers.yaml ids), so match the ids wherever they appear rather than assuming
    a prefix.
    """
    if isinstance(value, str):
        return ID_PATTERN.findall(value)
    if isinstance(value, dict):
        return [i for v in value.values() for i in collect_ids(v)]
    if isinstance(value, list):
        return [i for v in value for i in collect_ids(v)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contract", type=pathlib.Path, default=pathlib.Path("paper/contract.yaml"))
    parser.add_argument("--numbers", type=pathlib.Path, default=pathlib.Path("paper/evidence/numbers.yaml"))
    args = parser.parse_args()

    for path in (args.contract, args.numbers):
        if not path.exists():
            print(f"cannot run: {path} does not exist", file=sys.stderr)
            return 2

    entries = {e["id"]: e for e in yaml.safe_load(args.numbers.read_text())}
    contract = yaml.safe_load(args.contract.read_text()) or {}
    claims = contract.get("claim_ledger") or []
    if not claims:
        print("cannot run: contract.yaml has no claim_ledger rows", file=sys.stderr)
        return 2

    failures: list[str] = []
    caveat_report: list[str] = []

    for claim in claims:
        cid = claim.get("id", "<no id>")
        cited = dict.fromkeys(collect_ids(claim.get("evidence")))
        for nid in cited:
            entry = entries.get(nid)
            if entry is None:
                failures.append(f"{cid}: cites {nid}, which is not in {args.numbers}")
                continue
            status = entry.get("status")
            if status == "superseded":
                failures.append(
                    f"{cid}: cites {nid} (status=superseded). Cite {entry.get('superseded_by')} instead.\n"
                    f"      why: {entry.get('supersede_reason')}"
                )
            elif status == "caveated":
                if not str(claim.get("notes") or "").strip():
                    failures.append(
                        f"{cid}: cites {nid} (status=caveated) but its `notes` field is empty. "
                        "A caveated number may be cited only while its caveat is stated; record in "
                        "`notes` how the claim's wording carries it."
                    )
                caveat_report.append(
                    f"  [{cid}] {claim.get('claim', '')}\n"
                    f"    evidence {nid} ({entry.get('value')} {entry.get('unit')})\n"
                    f"    caveat:  {entry.get('supersede_reason')}\n"
                    f"    notes:   {claim.get('notes') or '(empty)'}"
                )

    if caveat_report:
        print(f"caveated evidence cited by {len(caveat_report)} claim/evidence pairs -- confirm each caveat is in the prose:\n")
        print("\n\n".join(caveat_report))
        print()
    sys.stdout.flush()

    if failures:
        print(f"FAIL ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"PASS: {len(claims)} claims, no superseded evidence, every caveated citation carries notes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
