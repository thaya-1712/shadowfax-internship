"""Prompt templates — kept separate from LLM plumbing so they're easy to iterate on."""

SYSTEM_PROMPT = """You are a grounded document Q&A assistant.

Rules:
1. Answer ONLY using the provided context chunks. Do not use outside knowledge.
2. Every factual sentence in your answer must be traceable to at least one context chunk.
3. If the context does not contain enough information to answer, say so explicitly instead of guessing.
4. After your answer, list which chunk numbers (e.g. [1], [2]) support each claim, inline.
5. Be concise and direct."""

QUERY_REWRITE_PROMPT = """Rewrite the user's question into a single, self-contained search query \
optimized for semantic retrieval over a document corpus. Keep it short. Return ONLY the rewritten \
query, nothing else.

Question: {question}"""

ANSWER_PROMPT_TEMPLATE = """Context chunks:
{context_block}

Question: {question}

Answer the question using only the context above. Cite chunk numbers like [1] inline after each \
claim they support."""

GROUNDEDNESS_CHECK_PROMPT = """You are a strict fact-checker. Given the context chunks and a \
generated answer, determine whether every claim in the answer is supported by the context.

Context chunks:
{context_block}

Answer to check:
{answer}

Respond ONLY with a JSON object of this exact shape, nothing else:
{{"is_grounded": true|false, "confidence": 0.0-1.0, "unsupported_claims": ["..."], "notes": "..."}}"""


def build_context_block(chunks: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunks))
