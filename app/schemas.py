"""
Typed request/response/domain models (Pydantic).
Every boundary in the system — API in, API out, and inter-component state — is validated here.
This is what "validation using typed schemas" in the brief refers to.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "md"


class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    doc_type: DocumentType
    num_chunks: int = 0
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    checksum: str


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    doc_type: DocumentType
    num_chunks: int
    message: str = "Document ingested successfully."


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total_chunks: int


# ---------------------------------------------------------------------------
# Chunking / retrieval internals
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    similarity_score: float
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Query / answer
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    doc_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these document IDs (document-scoped retrieval)."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    stream: bool = False

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    snippet: str


class GroundednessReport(BaseModel):
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    notes: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    groundedness: GroundednessReport
    rewritten_query: str | None = None
    latency_ms: int


# ---------------------------------------------------------------------------
# Workflow state (passed between LangGraph nodes)
# ---------------------------------------------------------------------------

class WorkflowState(BaseModel):
    """Single source of truth threaded through every graph node."""
    question: str
    doc_ids: list[str] | None = None
    top_k: int = 4

    rewritten_query: str | None = None
    candidates: list[RetrievedChunk] = Field(default_factory=list)
    reranked: list[RetrievedChunk] = Field(default_factory=list)

    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    groundedness: GroundednessReport | None = None

    class Config:
        arbitrary_types_allowed = True
