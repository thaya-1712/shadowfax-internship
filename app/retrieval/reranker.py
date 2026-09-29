"""
Reranking stage.

Uses a cross-encoder when available (much stronger relevance signal than
bi-encoder cosine similarity alone) and falls back to a lightweight lexical
overlap score if sentence-transformers' cross-encoder can't be loaded, so the
pipeline degrades gracefully rather than failing.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.schemas import RetrievedChunk


@lru_cache
def _get_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def _lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = set(re.findall(r"\w+", query.lower()))
    t_tokens = set(re.findall(r"\w+", text.lower()))
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()
    if cross_encoder is not None:
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = cross_encoder.predict(pairs)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
    else:
        for c in candidates:
            c.rerank_score = _lexical_overlap_score(query, c.chunk.text)

    ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    return ranked[:top_k]
