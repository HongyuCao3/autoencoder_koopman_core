"""Content-derived lifting feature for Koopman detection-design option 4
(docs/experiments/koopman_detection_design.md): a per-turn scalar measuring
how much a turn's text resembles a reference corpus of known attack-bank
queries, meant to be fed into ReducedStateConfig's `aux_cols` so it becomes
part of the Koopman state z_t instead of being invisible to the surrogate
the way raw text always is.

From-scratch TF-IDF + cosine similarity (no sklearn dependency) -- same
"don't add a dependency for a few dozen lines of pure numpy" rationale as
modeling.koopman.controllability_diagnostics's duplication note. This is
deliberately a minimal bag-of-words signature match, not semantic embedding
similarity: the question this feature exists to answer is whether even a
crude "does this look like known attack language" signal helps detection at
all, before reaching for anything heavier.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal fixed stopword list (not an NLTK download -- this project's
# vendored NLTK data only has vader_lexicon/punkt/tagger, see
# environment/setup_env.sh) covering function words common to BOTH attack
# and benign instruction-style phrasing ("how do I...", "please write...").
# IDF alone already down-weights terms that appear in nearly every reference
# document, but the reference corpus here is small (~20 attacks) and a
# handful of generic imperative words can still dominate a short query's
# cosine similarity even at reduced weight -- stripping them outright pushes
# the signal toward the actual content words.
_STOPWORDS = frozenset(
    """
    a an the this that these those is are was were be been being do does did
    doing have has had having i you he she it we they me him her us them my
    your his its our their and or but if so to of in on at for with about
    as by from into onto through please can could would should will shall
    just now then there here what which who whom how why when where not no
    yes very really quite also
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


@dataclass
class TfidfCorpus:
    vocab: dict[str, int]
    idf: np.ndarray
    doc_vectors: np.ndarray  # (n_docs, vocab_size), L2-normalized rows


def fit_tfidf_corpus(texts: list[str]) -> TfidfCorpus:
    """Fits vocabulary + smoothed IDF + normalized document vectors from
    `texts` (the reference corpus -- e.g. known attack queries). Frozen after
    fitting: query text is vectorized against this fixed vocab/IDF, new
    tokens in a query that never appeared in `texts` are simply dropped
    (out-of-vocabulary), not added."""

    if not texts:
        raise ValueError("empty reference corpus")
    tokenized = [tokenize(t) for t in texts]
    vocab: dict[str, int] = {}
    for tokens in tokenized:
        for tok in set(tokens):
            vocab.setdefault(tok, len(vocab))

    n_docs = len(texts)
    doc_freq = np.zeros(len(vocab))
    for tokens in tokenized:
        for tok in set(tokens):
            doc_freq[vocab[tok]] += 1
    idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1.0  # sklearn-style smoothed IDF

    doc_vectors = np.zeros((n_docs, len(vocab)))
    for i, tokens in enumerate(tokenized):
        for tok, count in Counter(tokens).items():
            doc_vectors[i, vocab[tok]] = count * idf[vocab[tok]]
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_vectors = doc_vectors / norms
    return TfidfCorpus(vocab=vocab, idf=idf, doc_vectors=doc_vectors)


def _vectorize_query(text: str, corpus: TfidfCorpus) -> np.ndarray:
    vec = np.zeros(len(corpus.vocab))
    for tok, count in Counter(tokenize(text)).items():
        idx = corpus.vocab.get(tok)
        if idx is not None:
            vec[idx] = count * corpus.idf[idx]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def max_similarity(text: str, corpus: TfidfCorpus) -> float:
    """Cosine similarity between `text` and its nearest neighbor among
    `corpus`'s reference documents. Text with no vocabulary overlap at all
    (empty string, or every token out-of-vocabulary) returns 0.0 rather than
    NaN -- "no evidence of resemblance" is a valid, fittable value."""

    if not text:
        return 0.0
    vec = _vectorize_query(text, corpus)
    if not np.any(vec):
        return 0.0
    return float(np.max(corpus.doc_vectors @ vec))


def annotate_similarity(
    rows: list[dict], text_col: str, corpus: TfidfCorpus, out_col: str = "attack_similarity"
) -> list[dict]:
    """Returns NEW row dicts (does not mutate `rows`) with `out_col` set to
    `max_similarity(row[text_col], corpus)` -- ready to pass straight into
    `ReducedStateConfig(aux_cols=(out_col,))`."""

    return [dict(row, **{out_col: max_similarity(row[text_col], corpus)}) for row in rows]


def reference_texts_excluding_ids(
    rows: list[dict], exclude_ids: set[str], text_col: str, id_col: str = "attack_id"
) -> list[str]:
    """Deduplicated reference corpus texts from `rows`, skipping any row
    whose `id_col` is in `exclude_ids` -- the standard way this module's
    callers build a leak-free "known attack signatures" corpus: exclude the
    held-out attacks a detector will later be evaluated on, so a held-out
    attack's near-identical replayed text isn't already sitting in its own
    reference corpus. Deduplicated on (id_col, text) so a fixed multi-turn
    attack replayed across several seeds doesn't get counted, and weighted
    in IDF, once per seed."""

    seen: set[tuple[str, str]] = set()
    texts: list[str] = []
    for row in rows:
        row_id = row[id_col]
        if row_id in exclude_ids:
            continue
        key = (row_id, row[text_col])
        if key in seen:
            continue
        seen.add(key)
        texts.append(row[text_col])
    return texts
