"""
Chunker — splits an input into per-chunk token spans for cache lookup.

Phase 2 deliverable. The chunker's hash is computed from chunk text only (not
position) so the same document at any position resolves to the same KV store
entry.

Tokenization convention: each chunk's text is tokenized with
``add_special_tokens=False``, and the fused input is the *concatenation* of
chunk token IDs — never a re-tokenization of the joined string. This avoids
BPE boundary drift (e.g. " hello" vs "hello") and matches what
``precompute_chunk_kv`` saw when it cached the chunk.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor


@dataclass
class Chunk:
    """A piece of input mapped to its absolute position in the fused sequence."""
    text: str
    token_ids: Tensor          # (S_chunk,), dtype int64
    position: int              # absolute start position in the fused sequence
    hash: str                  # text-based hash; position-independent
    is_cached: bool = False    # whether KVStore reported a hit at fuse time


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_texts(texts: Sequence[str], tokenizer) -> List[Chunk]:
    """Tokenize each chunk text independently, then assign positions.

    The returned chunks satisfy:
      ``torch.cat([c.token_ids for c in chunks])`` == fused sequence tokens.

    No BOS/EOS/pad is inserted — callers prepend a system prompt as a chunk if
    they want that role, and the framework treats it the same as any other
    chunk.
    """
    chunks: List[Chunk] = []
    pos = 0
    for text in texts:
        ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(torch.long)
        chunks.append(
            Chunk(
                text=text,
                token_ids=ids,
                position=pos,
                hash=_hash_text(text),
            )
        )
        pos += int(ids.shape[0])
    return chunks
