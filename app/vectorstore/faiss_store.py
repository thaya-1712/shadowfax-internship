"""
FAISS-backed vector index with a parallel metadata store.

FAISS only stores vectors + integer ids; the Chunk objects (text, doc_id,
filename, ...) live in a sidecar dict that is persisted alongside the index so
the whole store survives process restarts (containerized reproducibility).
"""
from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path

import faiss
import numpy as np

from app.schemas import Chunk, RetrievedChunk


class FaissVectorStore:
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._index_path = self.index_dir / "index.faiss"
        self._meta_path = self.index_dir / "meta.pkl"

        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "rb") as f:
                self._id_to_chunk: dict[int, Chunk] = pickle.load(f)
            self._next_id = max(self._id_to_chunk.keys(), default=-1) + 1
        else:
            # Inner product over normalized vectors == cosine similarity
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            self._id_to_chunk = {}
            self._next_id = 0

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks/vectors length mismatch")
        with self._lock:
            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype="int64")
            self._index.add_with_ids(vectors, ids)
            for i, chunk in zip(ids, chunks):
                self._id_to_chunk[int(i)] = chunk
            self._next_id += len(chunks)
            self._persist()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            # Over-fetch when scoping to specific docs, since FAISS itself is doc-agnostic
            fetch_k = top_k if not doc_ids else min(self._index.ntotal, top_k * 8)
            scores, ids = self._index.search(query_vector.reshape(1, -1), fetch_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self._id_to_chunk.get(int(idx))
            if chunk is None:
                continue
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            results.append(RetrievedChunk(chunk=chunk, similarity_score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def list_documents(self) -> dict[str, dict]:
        docs: dict[str, dict] = {}
        for chunk in self._id_to_chunk.values():
            entry = docs.setdefault(chunk.doc_id, {"filename": chunk.filename, "num_chunks": 0})
            entry["num_chunks"] += 1
        return docs

    def total_chunks(self) -> int:
        return len(self._id_to_chunk)

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._id_to_chunk, f)
