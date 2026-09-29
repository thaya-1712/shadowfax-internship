"""
Multi-format document loading.
Each loader returns raw text plus light structural metadata (e.g. page number)
so downstream chunking can preserve provenance.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.schemas import DocumentType


@dataclass
class LoadedDocument:
    filename: str
    doc_type: DocumentType
    text: str
    checksum: str
    page_map: list[tuple[int, int, int]] = field(default_factory=list)
    # page_map entries: (char_start, char_end, page_number) — empty for txt/md


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def load_pdf(path: Path) -> LoadedDocument:
    from pypdf import PdfReader

    raw = path.read_bytes()
    reader = PdfReader(str(path))
    text_parts: list[str] = []
    page_map: list[tuple[int, int, int]] = []
    cursor = 0
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if not page_text:
            continue
        start = cursor
        text_parts.append(page_text)
        cursor += len(page_text) + 2  # account for the "\n\n" join below
        page_map.append((start, cursor, i + 1))
    full_text = "\n\n".join(text_parts)
    return LoadedDocument(
        filename=path.name,
        doc_type=DocumentType.PDF,
        text=full_text,
        checksum=_checksum(raw),
        page_map=page_map,
    )


def load_txt(path: Path) -> LoadedDocument:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return LoadedDocument(
        filename=path.name,
        doc_type=DocumentType.TXT,
        text=text,
        checksum=_checksum(raw),
    )


def load_markdown(path: Path) -> LoadedDocument:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return LoadedDocument(
        filename=path.name,
        doc_type=DocumentType.MARKDOWN,
        text=text,
        checksum=_checksum(raw),
    )


_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
}


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(_LOADERS)}")
    doc = loader(path)
    if not doc.text.strip():
        raise ValueError(f"No extractable text found in {path.name}")
    return doc
