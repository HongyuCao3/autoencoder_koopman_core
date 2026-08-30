"""Free, no-extra-LLM-call surface features computed from already-generated
text -- a lightweight subset of the ~50-dim feature set from Alhafni et al.
2024 (docs/ALHAFNI_LINGUISTIC_CONTROL_FEASIBILITY.md), using only what the
pilot conda env already has installed (nltk's POS tagger + VADER lexicon,
per environment/setup_env.sh's nltk.downloader step) -- no spaCy, no
rstfinder. This is a deliberately small subset (structural counts + four
coarse POS-frequency ratios + VADER sentiment), not a full replication of
their feature set; see the feasibility doc for why the heavier parts
(dependency relations, RST discourse relations, FKGL) were left out here.
"""

from __future__ import annotations

from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

_ADJ_TAGS = {"JJ", "JJR", "JJS"}
_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
_VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
_ADV_TAGS = {"RB", "RBR", "RBS"}

SURFACE_FEATURE_NAMES = (
    "num_tokens",
    "num_sents",
    "avg_word_len",
    "ttr",
    "adj_ratio",
    "noun_ratio",
    "verb_ratio",
    "adv_ratio",
    "vader_sentiment",
)

_sia = SentimentIntensityAnalyzer()


def extract_surface_features(text: str) -> dict[str, float]:
    """Compute SURFACE_FEATURE_NAMES for one piece of text. NaN-safe: empty
    or punctuation-only text yields NaN for the ratio/length features
    (division by zero would otherwise be ambiguous, not zero)."""
    tokens = word_tokenize(text)
    words = [t for t in tokens if any(c.isalpha() for c in t)]
    num_tokens = len(tokens)
    num_sents = len(sent_tokenize(text)) if text.strip() else 0
    vader_sentiment = float(_sia.polarity_scores(text)["compound"]) if text.strip() else float("nan")

    if not words:
        return {
            "num_tokens": float(num_tokens),
            "num_sents": float(num_sents),
            "avg_word_len": float("nan"),
            "ttr": float("nan"),
            "adj_ratio": float("nan"),
            "noun_ratio": float("nan"),
            "verb_ratio": float("nan"),
            "adv_ratio": float("nan"),
            "vader_sentiment": vader_sentiment,
        }

    avg_word_len = sum(len(w) for w in words) / len(words)
    ttr = len(set(w.lower() for w in words)) / len(words)

    tags = [tag for _, tag in pos_tag(tokens)]
    n_tags = len(tags)
    adj_ratio = sum(1 for t in tags if t in _ADJ_TAGS) / n_tags
    noun_ratio = sum(1 for t in tags if t in _NOUN_TAGS) / n_tags
    verb_ratio = sum(1 for t in tags if t in _VERB_TAGS) / n_tags
    adv_ratio = sum(1 for t in tags if t in _ADV_TAGS) / n_tags

    return {
        "num_tokens": float(num_tokens),
        "num_sents": float(num_sents),
        "avg_word_len": float(avg_word_len),
        "ttr": float(ttr),
        "adj_ratio": float(adj_ratio),
        "noun_ratio": float(noun_ratio),
        "verb_ratio": float(verb_ratio),
        "adv_ratio": float(adv_ratio),
        "vader_sentiment": vader_sentiment,
    }
