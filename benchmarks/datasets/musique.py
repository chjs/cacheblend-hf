"""
Musique dataset loader.

The HuggingFace mirror ``dgslibisey/MuSiQue`` exposes
``{id, paragraphs, question, question_decomposition, answer, answer_aliases,
answerable}``. Each example has ~20 paragraphs (mix of supporting and
distractor), and a short answer (≤ 5 words typically).

We normalize each example to the project-wide schema:

    {
      "id":        str,
      "system":    str,    # short instruction prompt
      "documents": list[str],  # paragraph_text strings, in their original order
      "query":     str,    # the question
      "answer":    str,    # gold answer
      "answer_aliases": list[str],  # additional acceptable answers
    }

Phase 5 follows the paper's setup: feed all 20 paragraphs (supporting +
distractor) as documents. We do not run our own retriever — that's a
non-goal per ``GOAL.md``. Tests compare methods on the same fixed set.
"""
from __future__ import annotations

from typing import List, Optional

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using the provided "
    "documents. Be concise — answer in 5 words or fewer."
)


def load(
    split: str = "validation",
    limit: Optional[int] = None,
    name: str = "dgslibisey/MuSiQue",
    answerable_only: bool = True,
) -> List[dict]:
    """Load Musique and convert to the project-wide RAG schema.

    Args:
      split: HF split. Validation has 2,417 examples.
      limit: cap on number of examples returned. ``None`` for all.
      name: HF dataset name (override for mirrors).
      answerable_only: if True, drop the few ``answerable=False`` rows so
        we don't confuse F1 measurement with "I don't know" responses.

    Returns:
      List of dicts with the standardized schema.
    """
    ds = load_dataset(name, split=split)
    out: List[dict] = []
    for ex in ds:
        if answerable_only and not ex.get("answerable", True):
            continue
        out.append(
            {
                "id": ex["id"],
                "system": SYSTEM_PROMPT,
                "documents": [p["paragraph_text"] for p in ex["paragraphs"]],
                "query": ex["question"],
                "answer": ex["answer"],
                "answer_aliases": list(ex.get("answer_aliases", []) or []),
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out
