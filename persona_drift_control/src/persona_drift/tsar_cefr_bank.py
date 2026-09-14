"""Item bank for the `tsar_cefr` line (readability-level regulation on the
TSAR 2025 shared task, docs/operational_plan_tsar_cefr_line_2026-09-13.md).

The data is NOT vendored. Both HuggingFace dataset cards carry no `license`
field, and every other bank in this repo (`resources/sequor/`,
`resources/ergo_gsm8k_sharded.jsonl`) was vendored only with an explicit
licence. The signed alternative (plan section 3, Phase 1 block) is to fetch by
pinned revision at run time and record that revision in the run config:
same provenance strength, no redistribution. `fetch()` therefore refuses to
read a cached file it cannot tie to the pinned sha.

Three properties of the files that shape this module:

- `text_id` is `<source id>-<target level>`, so the SAME source paragraph
  appears twice, once per target level. `source_id` is what trajectories pair
  on and what the 5-fold split (plan section 3, Phase 1 block item 2) must cut
  along -- cutting on `text_id` would put a paragraph's a2 row in the fitting
  fold and its b1 row in the reporting fold, and the operator would have seen
  the reporting text.
- `target_cefr` is LOWER case in trial (`a2`) and UPPER case in test (`A2`).
  Normalised here; nothing downstream should case-fold again.
- `reference` is one simplification per (source, target level) pair -- not two
  annotators, which is what the plan assumed before the data was read.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.request
from dataclasses import dataclass

# Pinned revisions. resources/tsar_cefr/README.md carries the same table; these
# are the authority for what a run actually read.
SPLITS = {
    "trial": {
        "repo": "cardiffnlp/TSAR2025_SharedTask_RCTS_Trial-Data",
        "filename": "tsar2025_trial.jsonl",
        "revision": "c4d590b66dbb5dbd31c8a158f068c0adcfbe0af3",
        "n_rows": 40,
        "n_sources": 20,
    },
    "test": {
        "repo": "cardiffnlp/TSAR2025_SharedTask_RCTS_Test-Data",
        "filename": "tsar2025_test.jsonl",
        "revision": "3f4a9d38f82eb9126c2a069f0bd009301fb3df28",
        "n_rows": 200,
        "n_sources": 100,
    },
}

TARGET_LEVELS = ("A2", "B1")
CACHE_DIR = pathlib.Path(__file__).resolve().parents[3] / "resources" / "tsar_cefr" / "data"


@dataclass(frozen=True)
class TsarItem:
    text_id: str
    source_id: str
    original: str
    target_cefr: str
    reference: str


def _parse_row(row: dict) -> TsarItem:
    text_id = row["text_id"]
    target = row["target_cefr"].upper()
    if target not in TARGET_LEVELS:
        raise ValueError(f"{text_id}: unexpected target_cefr {row['target_cefr']!r}")
    source_id, _, suffix = text_id.rpartition("-")
    if not source_id or suffix.upper() != target:
        raise ValueError(
            f"{text_id}: text_id is expected to be '<source id>-<target level>' "
            f"agreeing with target_cefr {target}; the 5-fold split pairs on source_id"
        )
    return TsarItem(
        text_id=text_id,
        source_id=source_id,
        original=row["original"],
        target_cefr=target,
        reference=row["reference"],
    )


def load_tsar_bank(path: pathlib.Path) -> list[TsarItem]:
    """Parse one split file. Validates at the boundary; nothing downstream re-checks."""
    with open(path, encoding="utf-8") as handle:
        items = [_parse_row(json.loads(line)) for line in handle if line.strip()]
    ids = [item.text_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate text_id")
    return items


def source_ids(items: list[TsarItem]) -> list[str]:
    """Distinct source paragraphs, in first-appearance order. The unit of pairing."""
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item.source_id, None)
    return list(seen)


def fetch(split: str, cache_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Download the split at its pinned revision, or reuse an identical cache.

    The cache is keyed by revision, so a re-pin cannot be silently served from
    a stale file -- the failure mode that would make `resources/tsar_cefr/README.md`
    describe data the run did not read.
    """
    spec = SPLITS[split]
    cache_dir = cache_dir or CACHE_DIR
    cached = cache_dir / f"{split}.{spec['revision'][:12]}.jsonl"
    if not cached.exists():
        url = (
            f"https://huggingface.co/datasets/{spec['repo']}/resolve/"
            f"{spec['revision']}/{spec['filename']}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        cached.write_bytes(payload)
    items = load_tsar_bank(cached)
    if len(items) != spec["n_rows"] or len(source_ids(items)) != spec["n_sources"]:
        raise ValueError(
            f"{cached}: expected {spec['n_rows']} rows / {spec['n_sources']} sources at "
            f"revision {spec['revision']}, read {len(items)} / {len(source_ids(items))}"
        )
    return cached


def content_sha256(path: pathlib.Path) -> str:
    """For the run config: what these rows actually were, independent of the pin."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
