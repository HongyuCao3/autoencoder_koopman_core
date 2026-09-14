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

CRITERION 1b WAS REVISED AFTER ITS SECOND RUN (2026-09-14, user decision;
docs/tsar_cefr_gt1_closure_review_2026-09-14.md). It required the reference to
read simpler than its source on 100% of the 40 pairs. That threshold was signed
without consulting the instrument's published precision: the TSAR 2025 official
evaluator reports weighted-F1 0.89 and RMSE 0.34 on its own test set, RMSE ~0.6
against human annotators. At a 2% per-pair inversion rate a 40-pair sweep is
clean only 45% of the time; at 10% it is clean 1.5% of the time. The criterion
was asking for an instrument that does not exist, which is a different question
from whether this instrument is good enough to control with. It is replaced by
two criteria derived from that published precision, NOT from this run:

    1b'   direction correct on >= MIN_SIMPLER_SHARE of pairs (F1 0.89 leaves
          room for 6 of 40 inversions; the threshold is set at that room)
    1b''  the paired mean of level_delta has a 95% bootstrap CI strictly below
          zero -- the correct statistical statement of "references are simpler"

UNLIKE THE CRITERION 1 REWRITE ABOVE, THIS REVISION LOOSENS THE GATE AND IT WAS
MADE AFTER THE DATA WAS SEEN. The only thing that defends it is that 0.85 comes
from outside this run and this run's 0.95 clears it by a wide margin rather than
by a hair. `03-b1`, the one pair the classifier genuinely misread, is NOT
dropped: it stays in the denominator and it is named in the report. Dropping it
would be selecting the sample by the result.

Criterion 1 is the one that matters. `defense` died because its readout had no
range -- the judge gave the same value to every arm, so nothing could be
separated, and four rounds of judge repair did not change that. Here the same
question is asked of the CEFR classifier BEFORE any trajectory is generated:
across the 20 trial paragraphs and their 40 references, does the expected
level take enough distinct values, and does it order source above reference --
on nearly every pair (1b') and in the mean (1b'')? A classifier that cannot say
a reference simplification is simpler than its source cannot measure a
controller either.

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
import random
import statistics
import sys

MIN_DISTINCT_EXPECTED_LEVELS = 10
LEVEL_RESOLUTION = 0.05             # see criterion 1; 1e-6 counted float noise as range
MIN_SIMPLER_SHARE = 0.85            # 1b'; was 1.0, revised 2026-09-14 against F1 0.89
MAX_SIMPLER_MEAN_CI_UPPER = 0.0     # 1b''; the CI upper bound must sit below zero
MIN_FKGL_AGREEMENT = 0.70
MIN_MEANING_SD = 0.03

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260914           # fixed so the interval is reproducible from the artifact

# A pair both readouts call unchanged is a property of the data, not of the
# instrument. It is REPORTED and still COUNTED -- excluding it would move the
# denominator by the result.
DATA_PROPERTY_LEVEL_TOL = 0.1
DATA_PROPERTY_FKGL_TOL = 0.2

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "persona_drift_control" / "outputs" / "tsar_cefr_instruments" / "gt1_report_calibrated.json"
DEFAULT_RESCORE_OUT = DEFAULT_OUT.with_name("gt1_report_revised.json")

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
    p.add_argument("--out-path", type=pathlib.Path, default=None)
    p.add_argument("--rescore-from", type=pathlib.Path, default=None,
                   help="re-score an existing report's rows under the current criteria; "
                        "loads no model and reads no data (2026-09-14 criterion revision)")
    args = p.parse_args()
    if args.out_path is None:
        args.out_path = DEFAULT_RESCORE_OUT if args.rescore_from else DEFAULT_OUT
    return args


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
    """1b': how often the direction is right, against the evaluator's own F1."""
    share = sum(1 for d in deltas if d < 0) / len(deltas)
    return {"name": "reference_simpler_share", "value": share,
            "threshold": MIN_SIMPLER_SHARE, "n_pairs": len(deltas),
            "n_wrong_direction": sum(1 for d in deltas if d >= 0),
            "passed": share >= MIN_SIMPLER_SHARE}


def _bootstrap_mean_ci(clusters: list[list[float]]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(clusters)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        drawn = [clusters[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean([v for c in drawn for v in c]))
    means.sort()
    return means[int(0.025 * BOOTSTRAP_DRAWS)], means[int(0.975 * BOOTSTRAP_DRAWS)]


def criterion_reference_simpler_ci(rows: list[dict]) -> dict:
    """1b'': "the references are simpler" said as an interval instead of a count.

    Resampled BY SOURCE TEXT, not by row. The 40 pairs are 20 paragraphs each
    written to two target levels, so two rows sharing a source are not
    independent draws; plan section 4.4 fixes the same unit for the executor
    authority check. The by-row interval is computed too and reported beside
    it, but it never decides -- if the two ever disagree the clustered one is
    the verdict and the gap is the finding.
    """
    by_source: dict[str, list[float]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row["level_delta"])
    clusters = list(by_source.values())
    lo, hi = _bootstrap_mean_ci(clusters)
    row_lo, row_hi = _bootstrap_mean_ci([[row["level_delta"]] for row in rows])
    deltas = [row["level_delta"] for row in rows]
    return {"name": "reference_simpler_mean_ci", "value": hi,
            "threshold": MAX_SIMPLER_MEAN_CI_UPPER,
            "mean": statistics.fmean(deltas), "ci95": [lo, hi],
            "resample_unit": "source_id", "n_sources": len(clusters),
            "n_pairs": len(deltas), "ci95_by_row": [row_lo, row_hi],
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "passed": hi < MAX_SIMPLER_MEAN_CI_UPPER}


def data_property_pairs(rows: list[dict]) -> dict:
    """Pairs where both readouts say nothing moved -- reported, never removed."""
    named = [row["text_id"] for row in rows
             if abs(row["level_delta"]) < DATA_PROPERTY_LEVEL_TOL
             and abs(row["fkgl_delta"]) < DATA_PROPERTY_FKGL_TOL]
    return {"text_ids": named, "n": len(named), "counted_in_denominator": True,
            "level_tol": DATA_PROPERTY_LEVEL_TOL, "fkgl_tol": DATA_PROPERTY_FKGL_TOL}


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


def score(rows: list[dict]) -> tuple[list[dict], str]:
    """The verdict, from rows alone. The only place the criteria are combined,
    so a re-score cannot drift from a fresh run."""
    all_levels = ([row["source_level_expected"] for row in
                   {r["source_id"]: r for r in rows}.values()]
                  + [row["reference_level_expected"] for row in rows])
    criteria = [
        criterion_distinct_levels(all_levels),
        criterion_reference_simpler([r["level_delta"] for r in rows]),
        criterion_reference_simpler_ci(rows),
        criterion_fkgl_agreement([r["level_delta"] for r in rows],
                                 [r["fkgl_delta"] for r in rows]),
        criterion_meaning_range([r["meaning_to_source"] for r in rows]),
    ]
    return criteria, "PASS" if all(c["passed"] for c in criteria) else "FAIL"


def rescore(args: argparse.Namespace) -> None:
    """Apply the current criteria to rows already on disk. Zero GPU, no model,
    no network: the rows are the previous run's, only the verdict is new."""
    previous = json.loads(args.rescore_from.read_text(encoding="utf-8"))
    rows = previous["rows"]
    criteria, verdict = score(rows)
    report = dict(previous)
    report.update({
        "verdict": verdict,
        "criteria": criteria,
        "data_property_pairs": data_property_pairs(rows),
        "rescored_from": str(args.rescore_from),
        "rescored_verdict_was": previous["verdict"],
        "criteria_revision": "1b -> 1b' + 1b'' (2026-09-14); "
                             "docs/tsar_cefr_gt1_closure_review_2026-09-14.md",
        "provenance_of_rescore": provenance({"rescore_of": str(args.rescore_from)}),
    })
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_verdict(verdict, criteria, report, args.out_path)


def report_verdict(verdict: str, criteria: list[dict], report: dict, out_path: pathlib.Path) -> None:
    print(f"G-T1 {verdict}  ({report['split']}: {report['n_rows']} rows, "
          f"{report['n_sources']} sources, T={report['temperature']})")
    for c in criteria:
        mark = "ok " if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['value']:.4f} (threshold {c['threshold']})")
    print(f"  -> {out_path}")


def main() -> None:
    args = parse_args()
    if args.rescore_from:
        if args.out_path.exists():
            raise SystemExit(f"refusing to overwrite existing {args.out_path}")
        rescore(args)
        return
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

    criteria, verdict = score(rows)

    report = {
        "gate": "G-T1", "line": "tsar_cefr", "verdict": verdict,
        "temperature": temperature, "temperature_source": temperature_source,
        "split": args.split, "n_rows": len(rows), "n_sources": len(originals),
        "criteria": criteria,
        "data_property_pairs": data_property_pairs(rows),
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

    report_verdict(verdict, criteria, report, args.out_path)


if __name__ == "__main__":
    main()
