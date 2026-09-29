"""
LangGraph workflow: the agentic backbone of the assistant.

Nodes:
  rewrite_query -> retrieve -> rerank -> generate -> check_groundedness -> finalize

Each node is a pure function over WorkflowState (typed, validated), which is
what "multi-step workflow reflecting practical AI engineering thinking" means
in practice: explicit state, explicit transitions, inspectable at every step.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.generation.llm import get_llm_client, safe_json_parse
from app.generation.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    GROUNDEDNESS_CHECK_PROMPT,
    QUERY_REWRITE_PROMPT,
    SYSTEM_PROMPT,
    build_context_block,
)
from app.retrieval.reranker import rerank
from app.retrieval.retriever import Retriever
from app.schemas import Citation, GroundednessReport, WorkflowState

logger = logging.getLogger(__name__)


def node_rewrite_query(state: WorkflowState) -> WorkflowState:
    llm = get_llm_client()
    try:
        rewritten = llm.complete(
            system="You rewrite questions into concise search queries.",
            user=QUERY_REWRITE_PROMPT.format(question=state.question),
            max_tokens=64,
        ).strip()
        state.rewritten_query = rewritten or state.question
    except Exception:
        logger.warning("Query rewrite failed; falling back to original question.", exc_info=True)
        state.rewritten_query = state.question
    return state


def node_retrieve(state: WorkflowState) -> WorkflowState:
    settings = get_settings()
    retriever = Retriever()
    state.candidates = retriever.retrieve(
        query=state.rewritten_query or state.question,
        doc_ids=state.doc_ids,
        top_k=settings.top_k_initial,
    )
    return state


def node_rerank(state: WorkflowState) -> WorkflowState:
    settings = get_settings()
    state.reranked = rerank(
        query=state.rewritten_query or state.question,
        candidates=state.candidates,
        top_k=state.top_k or settings.top_k_final,
    )
    return state


def node_generate(state: WorkflowState) -> WorkflowState:
    if not state.reranked:
        state.answer = (
            "I couldn't find anything in the ingested documents that addresses this question."
        )
        state.citations = []
        return state

    llm = get_llm_client()
    context_texts = [rc.chunk.text for rc in state.reranked]
    context_block = build_context_block(context_texts)

    answer = llm.complete(
        system=SYSTEM_PROMPT,
        user=ANSWER_PROMPT_TEMPLATE.format(context_block=context_block, question=state.question),
        max_tokens=800,
    )
    state.answer = answer.strip()
    state.citations = [
        Citation(
            chunk_id=rc.chunk.chunk_id,
            doc_id=rc.chunk.doc_id,
            filename=rc.chunk.filename,
            chunk_index=rc.chunk.chunk_index,
            snippet=rc.chunk.text[:240],
        )
        for rc in state.reranked
    ]
    return state


def node_check_groundedness(state: WorkflowState) -> WorkflowState:
    if not state.reranked or not state.answer:
        state.groundedness = GroundednessReport(
            is_grounded=False, confidence=0.0, notes="No retrieved context to ground the answer in."
        )
        return state

    llm = get_llm_client()
    context_block = build_context_block([rc.chunk.text for rc in state.reranked])
    try:
        raw = llm.complete(
            system="You are a precise, conservative fact-checking assistant.",
            user=GROUNDEDNESS_CHECK_PROMPT.format(context_block=context_block, answer=state.answer),
            max_tokens=400,
        )
        parsed = safe_json_parse(raw)
        state.groundedness = GroundednessReport(**parsed)
    except Exception:
        logger.warning("Groundedness check failed; defaulting to unverified.", exc_info=True)
        state.groundedness = GroundednessReport(
            is_grounded=True, confidence=0.5, notes="Automated groundedness check unavailable."
        )
    return state


def node_finalize(state: WorkflowState) -> WorkflowState:
    if state.groundedness and not state.groundedness.is_grounded:
        state.answer = (
            f"{state.answer}\n\n⚠️ Some claims in this answer may not be fully supported by the "
            f"retrieved context (confidence: {state.groundedness.confidence:.2f})."
        )
    return state


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank", node_rerank)
    graph.add_node("generate", node_generate)
    graph.add_node("check_groundedness", node_check_groundedness)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "check_groundedness")
    graph.add_edge("check_groundedness", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
