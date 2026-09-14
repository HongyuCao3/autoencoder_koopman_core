#!/usr/bin/env python3
"""G-T1, the `tsar_cefr` instrument gate. CPU only, no GPU job depends on it
except the one it admits.

    passed  ->  submit GPU-1, the open-loop random-excitation arm (main line:
                identification data). Plan section 4.
    failed  ->  swap the instrument ONCE (classifier to the fallback estimator,
                FKGL to Flesch); a second failure CLOSES `tsar_cefr`
                (.claude/global.md: an instrument is fixed once, never twice).
    fills   ->  appendix A10 block 1 (instrument range and agreement).

THE THREE CRITERIA WERE SIGNED BEFORE THE DATA WAS READ (plan section 3.2, as
corrected by the Phase 1 availability check on 2026-09-13). They are constants
below and this script does not accept them as arguments: a gate whose
threshold can be passed on the command line is not a gate.

CRITERION 1 WAS REWRITTEN AFTER ITS FIRST RUN (2026-09-13, user decision). It
counted distinct values rounded to 1e-6 and passed at 56 of 60 while 88% of
those values sat within 0.01 of an integer: it was counting float noise as
range, which is the very failure it exists to catch. It now counts at a
resolution a controller could act on (LEVEL_RESOLUTION) and reports the
near-integer share beside it. The rewrite makes the gate STRICTER -- the first
run fails under it -- which is the only direction a signed criterion may move
after seeing data.

Criterion 1 is the one that matters. `defense` died because its readout had no
range -- the judge gave the same value to every arm, so nothing could be
separated, and four rounds of judge repair did not change that. Here the same
question is asked of the CEFR classifier BEFORE any trajectory is generated:
across the 20 trial paragraphs and their 40 references, does the expected
level take enough distinct values, and does it order source above reference on
every single pair? A classifier that cannot say a reference simplification is
simpler than its source cannot measure a controller either.

Criterion 2 guards the other direction. A classifier can have range and still
be measuring something other than readability; FKGL is deterministic, has no
training set, and moves for grammatical reasons. Requiring the two to agree in
DIRECTION on at least 70% of pairs is a cheap check that the range is the
right range. FKGL's own value is never reported.

Criterion 3 is MeaningBERT's range, checked the same way. It enters the signed
primary as an admission threshold (plan section 5.4), so a collapsed
distribution would silently admit everything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

MIN_DISTINCT_EXPECTED_LEVELS = 10
LEVEL_RESOLUTION = 0.05             # see criterion 1; 1e-6 counted float noise as range
REQUIRED_SIMPLER_SHARE = 1.0        # every pair: reference below source
MIN_FKGL_AGREEMENT = 0.70
MIN_MEANING_SD = 0.03

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "persona_drift_control" / "outputs" / "tsar_cefr_instruments" / "gt1_report_calibrated.json"

sys.path.insert(0, str(REPO_ROOT / "persona_drift_control" / "src"))

from persona_drift import tsar_cefr_bank as bank            # noqa: E402
from persona_drift import tsar_cefr_readout as readout      # noqa: E402
from persona_drift.run_provenance import provenance         # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="trial", choices=sorted(bank.SPLITS))
    p.add_argument("--device", type=int, default=-1, help="-1 CPU, >=0 CUDA index")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=None,
                   help="control-readout temperature; default fits it by NLL on this split's references")
    p.add_argument("--out-path", type=pathlib.Path, default=DEFAULT_OUT)
    return p.parse_args()


def criterion_distinct_levels(levels: list[float]) -> dict:
    """Distinct values a controller could tell apart, plus how many hug an integer."""
    distinct = len({round(v / LEVEL_RESOLUTION) for v in levels})
    near_integer = sum(1 for v in levels if abs(v - round(v)) < 0.01) / len(levels)
    return {"name": "distinct_expected_levels", "value": distinct,
            "threshold": MIN_DISTINCT_EXPECTED_LEVELS,
            "resolution": LEVEL_RESOLUTION, "near_integer_share": near_integer,
            "effective_integer_levels": sorted({round(v) for v in levels}),
            "passed": distinct >= MIN_DISTINCT_EXPECTED_LEVELS}


def criterion_reference_simpler(deltas: list[float]) -> dict:
    share = sum(1 for d in deltas if d < 0) / len(deltas)
    return {"name": "reference_simpler_share", "value": share,
            "threshold": REQUIRED_SIMPLER_SHARE, "n_pairs": len(deltas),
            "passed": share >= REQUIRED_SIMPLER_SHARE}


def criterion_fkgl_agreement(level_deltas: list[float], fkgl_deltas: list[float]) -> dict:
    agree = sum(1 for a, b in zip(level_deltas, fkgl_deltas)
                if (a > 0) == (b > 0) and (a < 0) == (b < 0))
    share = agree / len(level_deltas)
    return {"name": "fkgl_direction_agreement", "value": share,
            "threshold": MIN_FKGL_AGREEMENT, "n_pairs": len(level_deltas),
            "passed": share >= MIN_FKGL_AGREEMENT}


def criterion_meaning_range(scores: list[float]) -> dict:
    sd = statistics.stdev(scores)
    return {"name": "meaning_sd", "value": sd, "threshold": MIN_MEANING_SD,
            "mean": statistics.fmean(scores), "passed": sd > MIN_MEANING_SD}


def main() -> None:
    args = parse_args()
    if args.out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out_path}")

    path = bank.fetch(args.split)
    items = bank.load_tsar_bank(path)
    probes = readout.CefrProbes(device=args.device, batch_size=args.batch_size)

    originals = bank.source_ids(items)
    original_text = {item.source_id: item.original for item in items}
    source_probs = probes.probabilities([original_text[sid] for sid in originals])
    reference_probs = probes.probabilities([item.reference for item in items])

    if args.temperature is None:
        # Gold = the level each reference was written to. One parameter, NLL.
        picked = [p[readout.official_pick(p)] for p in reference_probs]
        temperature = readout.fit_temperature(picked, [item.target_cefr for item in items])
        temperature_source = f"fitted by NLL on {len(items)} {args.split} references"
    else:
        temperature = args.temperature
        temperature_source = "given on the command line"

    original_levels = [readout.read_level(p, temperature) for p in source_probs]
    source_level = dict(zip(originals, (r.level_expected for r in original_levels)))
    source_fkgl = {sid: readout.fkgl(original_text[sid]) for sid in originals}

    reference_levels = [readout.read_level(p, temperature) for p in reference_probs]
    meaning = probes.meaning([item.original for item in items],
                             [item.reference for item in items])

    rows = []
    for item, ref_level, meaning_score in zip(items, reference_levels, meaning):
        rows.append({
            "text_id": item.text_id, "source_id": item.source_id,
            "target_cefr": item.target_cefr,
            "source_level_expected": source_level[item.source_id],
            "reference_level_expected": ref_level.level_expected,
            "reference_level_official": ref_level.level_official,
            "level_delta": ref_level.level_expected - source_level[item.source_id],
            "fkgl_delta": readout.fkgl(item.reference) - source_fkgl[item.source_id],
            "meaning_to_source": meaning_score,
        })

    expected = bank.SPLITS[args.split]["n_rows"]
    if len(rows) != expected:
        raise SystemExit(f"{args.split}: filled {len(rows)} of {expected} rows; gate not decidable")

    all_levels = list(source_level.values()) + [r["reference_level_expected"] for r in rows]
    criteria = [
        criterion_distinct_levels(all_levels),
        criterion_reference_simpler([r["level_delta"] for r in rows]),
        criterion_fkgl_agreement([r["level_delta"] for r in rows],
                                 [r["fkgl_delta"] for r in rows]),
        criterion_meaning_range([r["meaning_to_source"] for r in rows]),
    ]
    verdict = "PASS" if all(c["passed"] for c in criteria) else "FAIL"

    report = {
        "gate": "G-T1", "line": "tsar_cefr", "verdict": verdict,
        "temperature": temperature, "temperature_source": temperature_source,
        "split": args.split, "n_rows": len(rows), "n_sources": len(originals),
        "criteria": criteria,
        "on_pass": "submit GPU-1 (open-loop random-excitation arm)",
        "on_fail": "swap the instrument once; a second failure closes tsar_cefr",
        "fills": "appendix A10 block 1",
        "data": {"split": args.split, "path": str(path),
                 "revision": bank.SPLITS[args.split]["revision"],
                 "content_sha256": bank.content_sha256(path)},
        "provenance": provenance({
            "cefr_models": list(readout.CEFR_MODELS),
            "meaning_model": readout.MEANING_MODEL,
            "device": args.device,
            "temperature": temperature,
        }),
        "rows": rows,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"G-T1 {verdict}  ({args.split}: {len(rows)} rows, {len(originals)} sources, T={temperature})")
    for c in criteria:
        mark = "ok " if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['value']:.4f} (threshold {c['threshold']})")
    print(f"  -> {args.out_path}")


if __name__ == "__main__":
    main()
