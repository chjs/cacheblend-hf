"""
RAG input builder — turns a dataset item into the chunk schema we feed to
``fuse_*``.

The chunker (``cacheblend.chunker.chunk_texts``) tokenizes each chunk text
independently with ``add_special_tokens=False`` and concatenates the resulting
ids, so the document order here is exactly the order the model sees in the
fused sequence. We split each document at ``chunk_size`` tokens of the model's
tokenizer; documents shorter than ``chunk_size`` become a single chunk.

The first chunk is always the system prompt, the last chunk is always the
query — that way a "prefix caching" baseline can hit the system prompt slot
deterministically and the query stays as the only uncached suffix on every
request.

Tokenizer-mismatch instrumentation (``count_tokenizer_mismatches``) measures
how often re-tokenizing the joined string would yield a different token count
than concatenating the per-chunk tokenizations. The Phase 5 report references
this number to characterize the BPE-boundary risk on real RAG data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


@dataclass
class RAGInput:
    """Assembled chunks: system prompt first, query last, documents in order."""
    chunk_texts: List[str]
    answer: str
    answer_aliases: List[str]
    item_id: str

    def joined_text(self) -> str:
        return "".join(self.chunk_texts)


def _split_text_by_tokens(text: str, tokenizer, chunk_size: int) -> List[str]:
    """Token-budgeted split. Returns one or more substrings of ``text`` whose
    independent tokenizations sum to at most ``chunk_size`` tokens each."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if int(ids.shape[0]) <= chunk_size:
        return [text]
    out: List[str] = []
    for start in range(0, int(ids.shape[0]), chunk_size):
        slice_ids = ids[start : start + chunk_size]
        out.append(tokenizer.decode(slice_ids))
    return out


def build_rag_input(
    item: dict,
    tokenizer,
    chunk_size: int = 512,
    *,
    query_template: str = "\n\nQuestion: {query}\nAnswer:",
) -> RAGInput:
    """Build a chunk list out of a standardized dataset item.

    The output list is: ``[system, doc_chunk_1, ..., doc_chunk_N, query]``.

    ``system`` and the formatted ``query`` are intentionally separate chunks
    so that:
      - The system prompt's KV is identical across requests (cache-friendly).
      - The query's KV is per-request (always uncached).
      - Documents in the middle are the candidates for chunk-level reuse.
    """
    chunks: List[str] = [item["system"] + "\n\n"]
    for doc in item["documents"]:
        chunks.extend(_split_text_by_tokens(doc + "\n\n", tokenizer, chunk_size))
    chunks.append(query_template.format(query=item["query"]))

    return RAGInput(
        chunk_texts=chunks,
        answer=item["answer"],
        answer_aliases=item.get("answer_aliases", []),
        item_id=item.get("id", ""),
    )


def count_tokenizer_mismatches(
    rag_inputs: Iterable[RAGInput],
    tokenizer,
) -> dict:
    """How often does re-tokenizing the joined string differ in length from
    concatenating per-chunk tokenizations?

    Returns ``{n, mismatches, mismatch_rate, mean_token_delta}``.
    """
    n = 0
    mismatches = 0
    deltas: List[int] = []
    for r in rag_inputs:
        n += 1
        per_chunk = sum(
            int(
                tokenizer(c, add_special_tokens=False, return_tensors="pt")
                .input_ids.shape[1]
            )
            for c in r.chunk_texts
        )
        joined_len = int(
            tokenizer(
                r.joined_text(), add_special_tokens=False, return_tensors="pt"
            ).input_ids.shape[1]
        )
        delta = abs(per_chunk - joined_len)
        deltas.append(delta)
        if delta != 0:
            mismatches += 1
    return {
        "n": n,
        "mismatches": mismatches,
        "mismatch_rate": (mismatches / n) if n else 0.0,
        "mean_token_delta": (sum(deltas) / n) if n else 0.0,
    }
