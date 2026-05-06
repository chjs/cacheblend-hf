"""
SQuAD-style token-level F1 for short-answer QA.

Convention follows the paper's Musique / 2WikiMQA evaluation: lowercase, strip
punctuation, drop articles ("a", "an", "the"), collapse whitespace, then
tokenize on whitespace and compute the token-overlap F1 between prediction
and gold. When the gold has multiple acceptable forms (Musique's
``answer_aliases``) take the max F1 across them.
"""
from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from typing import Iterable, List, Sequence


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TRANS)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def _f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(pred: str, gold: str | Sequence[str]) -> float:
    """Token-level F1; takes max across alias list when ``gold`` is a sequence."""
    if isinstance(gold, str):
        return _f1(pred, gold)
    return max((_f1(pred, g) for g in gold), default=0.0)


def aggregate(scores: Iterable[float]) -> dict:
    xs: List[float] = list(scores)
    if not xs:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    xs_sorted = sorted(xs)
    p95_idx = max(0, int(round(0.95 * (len(xs_sorted) - 1))))
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "p50": statistics.median(xs_sorted),
        "p95": xs_sorted[p95_idx],
    }
