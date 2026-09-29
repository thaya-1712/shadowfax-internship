"""
Chunking strategy: recursive character splitting with overlap.

Splits on the largest available semantic boundary first (blank line -> line ->
sentence -> word -> character) so chunks stay coherent instead of cutting
mid-sentence, while respecting a hard character budget.
"""
from __future__ import annotations

import re
import uuid

from app.config import get_settings
from app.ingestion.loaders import LoadedDocument
from app.schemas import Chunk

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    def _recursive_split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size:
            return [t] if t.strip() else []
        if not seps:
            # hard cut as last resort
            return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]

        sep, rest_seps = seps[0], seps[1:]
        parts = t.split(sep) if sep else list(t)

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_recursive_split(part, rest_seps))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _recursive_split(text, _SEPARATORS)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + chunk)
    return overlapped


def chunk_document(doc: LoadedDocument, doc_id: str) -> list[Chunk]:
    settings = get_settings()
    pieces = _split_text(doc.text, settings.chunk_size, settings.chunk_overlap)

    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        piece = re.sub(r"[ \t]+", " ", piece).strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                filename=doc.filename,
                text=piece,
                chunk_index=idx,
                metadata={"doc_type": doc.doc_type.value},
            )
        )
    return chunks
