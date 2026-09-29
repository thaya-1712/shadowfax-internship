"""
FastAPI entrypoint.

Endpoints:
  POST /ingest    -- upload a PDF/TXT/MD file, chunk + embed + index it
  POST /query     -- ask a grounded question (runs the LangGraph workflow)
  GET  /documents -- list ingested documents
  GET  /health    -- liveness probe
"""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import load_document
from app.retrieval.retriever import get_vector_store
from app.schemas import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentType,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.utils.logging import configure_logging
from app.workflow.graph import get_compiled_graph
from app.schemas import WorkflowState

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Production-style Retrieval-Augmented Generation assistant over PDF/TXT/Markdown.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {_SUPPORTED_SUFFIXES}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4().hex[:12]
    dest_path = upload_dir / f"{doc_id}_{file.filename}"

    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loaded = load_document(dest_path)
        chunks = chunk_document(loaded, doc_id=doc_id)
        if not chunks:
            raise HTTPException(422, "Document produced no usable chunks after processing.")

        embedder = get_embedder()
        vectors = embedder.embed_texts([c.text for c in chunks])

        store = get_vector_store()
        store.add(chunks, vectors)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc

    doc_type = {
        ".pdf": DocumentType.PDF,
        ".txt": DocumentType.TXT,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
    }[suffix]

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename or dest_path.name,
        doc_type=doc_type,
        num_chunks=len(chunks),
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    store = get_vector_store()
    docs = store.list_documents()
    metadata = [
        DocumentMetadata(
            doc_id=doc_id,
            filename=info["filename"],
            doc_type=DocumentType.PDF if info["filename"].endswith(".pdf") else (
                DocumentType.MARKDOWN if info["filename"].endswith((".md", ".markdown")) else DocumentType.TXT
            ),
            num_chunks=info["num_chunks"],
            checksum="",
        )
        for doc_id, info in docs.items()
    ]
    return DocumentListResponse(documents=metadata, total_chunks=store.total_chunks())


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    store = get_vector_store()
    if store.total_chunks() == 0:
        raise HTTPException(400, "No documents have been ingested yet. Call /ingest first.")

    start = time.perf_counter()
    graph = get_compiled_graph()

    initial_state = WorkflowState(
        question=request.question,
        doc_ids=request.doc_ids,
        top_k=request.top_k or settings.top_k_final,
    )
    result = graph.invoke(initial_state)
    # LangGraph returns a dict-like state; normalize back into our schema
    final_state = WorkflowState(**result) if not isinstance(result, WorkflowState) else result

    latency_ms = int((time.perf_counter() - start) * 1000)

    return QueryResponse(
        question=request.question,
        answer=final_state.answer or "",
        citations=final_state.citations,
        retrieved_chunks=final_state.reranked,
        groundedness=final_state.groundedness,
        rewritten_query=final_state.rewritten_query,
        latency_ms=latency_ms,
    )
