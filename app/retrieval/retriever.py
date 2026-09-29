"""
Retrieval orchestration: embeds the query and pulls candidate chunks from
the FAISS store, optionally scoped to specific document ids.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.schemas import RetrievedChunk
from app.vectorstore.faiss_store import FaissVectorStore


@lru_cache
def get_vector_store() -> FaissVectorStore:
    settings = get_settings()
    embedder = get_embedder()
    return FaissVectorStore(dim=embedder.dim, index_dir=settings.index_dir)


class Retriever:
    def __init__(self):
        self.settings = get_settings()
        self.embedder = get_embedder()
        self.store = get_vector_store()

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k_initial
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, top_k=top_k, doc_ids=doc_ids)
        # Drop weak matches early — cheap groundedness safeguard before generation.
        return [r for r in results if r.similarity_score >= self.settings.min_similarity_score]
