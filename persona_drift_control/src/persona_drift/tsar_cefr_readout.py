"""Readouts for the `tsar_cefr` line: CEFR level, meaning preservation, FKGL.

TWO LAYERS, AND THEY MUST NOT BE SWAPPED (.claude/global.md -> selection vs
reporting; plan section 3, Phase 1 block item 3).

- `level_expected` -- the CONTROL readout. sum_k k*p_k over the six CEFR
  levels, A1=1..C2=6. The controller optimises this because it has range;
  the official argmax has six values, and a readout with no range is exactly
  how the `defense` line died (its judge could not separate arms).
- `level_official` -- the REPORT readout. The shared task's own evaluator:
  run all three ModernBERT classifiers, keep the one whose top-1 confidence
  is highest, take ITS argmax (code/tsar2025_evaluation_script.py:
  `best = max((top1(d1), top1(d2), top1(d3)), key=lambda d: d["score"])`).
  Every reported CEFR number comes from here, so that "on TSAR 2025" means
  what the shared task means by it.

The expectation is taken over the probabilities of the SAME classifier the
official rule selected, not over a pooled or averaged distribution: one
selection rule, two functionals of its output. Averaging the three would make
the control readout answer a question the reported one never asks.

TEMPERATURE. G-T1's first run (2026-09-13, gt1_range_diagnostic.json) found the
selected classifier near one-hot: 88% of readouts within 0.01 of an integer, so
the expectation collapsed onto the argmax and the control readout had four
effective values. The signed instrument swap (one only, .claude/global.md) is
to soften the distribution, not to change the model: p_T is proportional to
p**(1/T), which is exactly dividing the logits by T since softmax ignores an
additive constant. T is fitted by NLL against the gold target level of trial
references -- a one-parameter calibration with a proper objective, NOT tuned to
make the gate pass. `level_official` is computed at T=1 throughout: the report
layer stays the shared task's own evaluator.

FKGL is implemented here rather than taken from `textstat` -- the shared
environment is also the one the GPU arms run in, and .claude/experiments.md
records two jobs killed by concurrent installs into it. The syllable rule is
the usual vowel-group heuristic; it is a DIRECTION check against
`level_expected` (gate G-T1 criterion 2), never a reported number.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# The shared task's evaluator, pinned. resources/tsar_cefr/README.md carries the shas.
CEFR_MODELS = (
    "AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
    "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
    "AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2",
)
MEANING_MODEL = "davebulaval/MeaningBERT"

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE = re.compile(r"[.!?]+(?:\s|$)")
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


@dataclass(frozen=True)
class LevelReadout:
    level_expected: float
    level_official: str
    official_model: str
    confidence: float


def official_pick(per_model_probs: Sequence[Sequence[float]]) -> int:
    """Index of the classifier the shared task's rule keeps: highest top-1 probability."""
    if not per_model_probs:
        raise ValueError("no classifier outputs to pick from")
    for probs in per_model_probs:
        if len(probs) != len(CEFR_LEVELS):
            raise ValueError(
                f"expected {len(CEFR_LEVELS)} CEFR probabilities, got {len(probs)}"
            )
    return max(range(len(per_model_probs)), key=lambda i: max(per_model_probs[i]))


def expected_level(probs: Sequence[float]) -> float:
    """sum_k k*p_k with A1=1..C2=6. The control readout's range comes from here."""
    if len(probs) != len(CEFR_LEVELS):
        raise ValueError(f"expected {len(CEFR_LEVELS)} probabilities, got {len(probs)}")
    return sum((k + 1) * p for k, p in enumerate(probs))


def temperature_scale(probs: Sequence[float], temperature: float) -> list[float]:
    """p_T proportional to p**(1/T). T=1 is the identity; T>1 softens."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return list(probs)
    floored = [max(p, 1e-12) for p in probs]
    scaled = [math.exp(math.log(p) / temperature) for p in floored]
    total = sum(scaled)
    return [v / total for v in scaled]


def fit_temperature(
    per_text_probs: Sequence[Sequence[float]],
    gold_levels: Sequence[str],
    grid: Sequence[float] | None = None,
) -> float:
    """The T minimising NLL of the gold level. One parameter, proper objective.

    Deliberately NOT a search over anything the gate measures: a temperature
    chosen to maximise the readout's spread would be selecting on the gate.
    """
    if len(per_text_probs) != len(gold_levels):
        raise ValueError("one gold level per text is required")
    indices = [CEFR_LEVELS.index(level) for level in gold_levels]
    grid = grid or [round(0.1 * i, 2) for i in range(1, 201)]

    def nll(temperature: float) -> float:
        total = 0.0
        for probs, index in zip(per_text_probs, indices):
            total -= math.log(max(temperature_scale(probs, temperature)[index], 1e-12))
        return total

    return min(grid, key=nll)


def read_level(
    per_model_probs: Sequence[Sequence[float]], temperature: float = 1.0
) -> LevelReadout:
    """Both layers from one selection. `per_model_probs` is ordered like CEFR_MODELS.

    `temperature` softens the CONTROL layer only; the official label and the
    selection itself are computed on the untouched distribution.
    """
    index = official_pick(per_model_probs)
    probs = per_model_probs[index]
    argmax = max(range(len(probs)), key=lambda k: probs[k])
    return LevelReadout(
        level_expected=expected_level(temperature_scale(probs, temperature)),
        level_official=CEFR_LEVELS[argmax],
        official_model=CEFR_MODELS[index] if index < len(CEFR_MODELS) else str(index),
        confidence=probs[argmax],
    )


def count_syllables(word: str) -> int:
    """Vowel groups, with the silent final `e` dropped. Minimum one."""
    lowered = word.lower()
    groups = len(_VOWEL_GROUP.findall(lowered))
    if lowered.endswith("e") and not lowered.endswith(("le", "ee", "ye")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def fkgl(text: str) -> float:
    """Flesch-Kincaid grade level. Direction check only -- never a reported number."""
    words = _WORD.findall(text)
    if not words:
        raise ValueError("FKGL is undefined on a text with no words")
    sentences = max(len(_SENTENCE.findall(text.strip())), 1)
    syllables = sum(count_syllables(word) for word in words)
    return 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59


class CefrProbes:
    """The three classifiers plus MeaningBERT, loaded once. CPU by default.

    Kept out of the pure functions above so the gate's unit tests never need a
    model download: the arithmetic that decides G-T1 is tested on constructed
    probability vectors, the loading is exercised only by the real run.
    """

    def __init__(self, device: int = -1, batch_size: int = 16) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        self.batch_size = batch_size
        self._pipelines = []
        for name in CEFR_MODELS:
            pipe = pipeline("text-classification", model=name, device=device, top_k=None)
            labels = [pipe.model.config.id2label[i] for i in range(len(CEFR_LEVELS))]
            if tuple(labels) != CEFR_LEVELS:
                raise ValueError(f"{name}: label set is {labels}, expected {CEFR_LEVELS}")
            self._pipelines.append(pipe)
        self._meaning_tokenizer = AutoTokenizer.from_pretrained(MEANING_MODEL)
        self._meaning_model = AutoModelForSequenceClassification.from_pretrained(MEANING_MODEL)
        self._meaning_model.eval()

    def probabilities(self, texts: Sequence[str]) -> list[list[list[float]]]:
        """Per text, the three classifiers' CEFR distributions, ordered like CEFR_MODELS."""
        per_model = []
        for pipe in self._pipelines:
            outputs = pipe(list(texts), batch_size=self.batch_size)
            per_model.append([_ordered_probs(entry) for entry in outputs])
        return [[per_model[m][i] for m in range(len(per_model))] for i in range(len(texts))]

    def levels(self, texts: Sequence[str], temperature: float = 1.0) -> list[LevelReadout]:
        return [read_level(p, temperature) for p in self.probabilities(texts)]

    def meaning(self, sources: Sequence[str], targets: Sequence[str]) -> list[float]:
        """MeaningBERT similarity in [0, 1]; the official script divides its head by 100."""
        import torch

        scores: list[float] = []
        for start in range(0, len(sources), self.batch_size):
            batch_src = list(sources[start:start + self.batch_size])
            batch_tgt = list(targets[start:start + self.batch_size])
            encoded = self._meaning_tokenizer(
                batch_src, batch_tgt, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                logits = self._meaning_model(**encoded).logits
            scores.extend(float(v) / 100.0 for v in logits.squeeze(-1).tolist())
        return scores


def _ordered_probs(entry) -> list[float]:
    by_label = {item["label"]: float(item["score"]) for item in entry}
    missing = set(CEFR_LEVELS) - set(by_label)
    if missing:
        raise ValueError(f"classifier returned no score for {sorted(missing)}")
    return [by_label[level] for level in CEFR_LEVELS]
