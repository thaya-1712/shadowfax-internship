# ShadowFox RAG Assistant (Advanced Level)

A production-style Retrieval-Augmented Generation assistant for grounded Q&A over
PDF, TXT, and Markdown documents — built with FastAPI, Streamlit, Pydantic,
LangGraph, and FAISS.

## Architecture

```
Upload → Load → Chunk → Embed → Index (FAISS)
                                        │
Question → Rewrite → Retrieve → Rerank → Generate → Groundedness Check → Answer + Citations
           (LangGraph state machine, typed at every edge)
```

**Why this shape:**
- **Ingestion is decoupled from querying.** `app/ingestion/` only ever produces
  `Chunk` objects; `app/vectorstore/` only ever stores/searches vectors. Neither
  knows about the LLM. This separation is what lets you swap the embedding
  model or the vector store without touching retrieval or generation.
- **LangGraph gives the pipeline an explicit, inspectable state machine**
  (`app/workflow/graph.py`) instead of one long function. Each node
  (`rewrite_query`, `retrieve`, `rerank`, `generate`, `check_groundedness`,
  `finalize`) reads and writes a single typed `WorkflowState`, so you can
  trace exactly what happened at each step, or swap/add a node (e.g. a second
  retrieval pass) without touching the rest.
- **Every boundary is a Pydantic model** (`app/schemas.py`): API requests,
  API responses, chunks, retrieved chunks, and the workflow state itself. This
  is what catches malformed input/output early instead of failing deep inside
  a prompt string.
- **Hallucination control is two-layered, not just "hope the prompt works":**
  1. The generation prompt forces inline citations (`[1]`, `[2]`) tied to
     specific chunks.
  2. A second LLM call (`check_groundedness`) independently re-reads the
     answer against the same context and flags unsupported claims — an
     explicit self-verification step, surfaced to the user as a confidence
     score, not silently swallowed.
- **Reranking** re-scores FAISS's initial candidates with a cross-encoder
  (falls back to lexical overlap if the cross-encoder can't load), because
  bi-encoder cosine similarity alone is a weak relevance signal — this
  materially improves which chunks reach the LLM.
- **Document-scoped retrieval:** `doc_ids` can be passed on `/query` to
  restrict search to specific ingested files, letting the same index serve
  multiple documents without cross-contaminating answers.
- **Containerized reproducibility:** `docker-compose.yml` runs the API and
  Streamlit UI as separate services sharing a `data/` volume, so the FAISS
  index and uploaded files persist across restarts.

## Project layout

```
app/
  config.py          # all tunables (chunk size, top_k, model names, provider)
  schemas.py          # every Pydantic model used across the system
  ingestion/
    loaders.py        # PDF / TXT / Markdown -> raw text
    chunking.py        # recursive character splitter with overlap
  embeddings/
    embedder.py         # sentence-transformers wrapper
  vectorstore/
    faiss_store.py      # FAISS index + metadata persistence
  retrieval/
    retriever.py         # query embed + FAISS search + doc scoping
    reranker.py           # cross-encoder (or lexical fallback) rerank
  generation/
    prompts.py             # all prompt templates
    llm.py                   # Anthropic / OpenAI client abstraction
  workflow/
    graph.py                  # LangGraph state machine wiring it all together
  main.py                       # FastAPI app (/ingest, /query, /documents, /health)
streamlit_app.py                # chat UI, talks to the API only over HTTP
Dockerfile / docker-compose.yml
requirements.txt
.env.example
```

## Setup

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY (or switch LLM_PROVIDER=openai and set OPENAI_API_KEY)

pip install -r requirements.txt
```

## Running locally

```bash
# terminal 1 — API
uvicorn app.main:app --reload --port 8000

# terminal 2 — UI
streamlit run streamlit_app.py
```

Open http://localhost:8501, upload a PDF/TXT/MD file in the sidebar, then ask questions.

## Running with Docker

```bash
docker compose up --build
```

- API: http://localhost:8000/docs (interactive Swagger UI)
- UI: http://localhost:8501

## API reference

| Method | Endpoint      | Description                                   |
|--------|---------------|------------------------------------------------|
| POST   | `/ingest`     | Upload a PDF/TXT/MD file to chunk + index      |
| POST   | `/query`      | Ask a question; runs the full LangGraph flow   |
| GET    | `/documents`  | List ingested documents and chunk counts       |
| GET    | `/health`     | Liveness check                                 |

Example query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about onboarding?"}'
```

## Notes on reliability / observability

- Every retrieved chunk is returned in the API response alongside its
  similarity and rerank scores — nothing is hidden from the caller.
- The groundedness check's confidence score and any flagged unsupported
  claims are surfaced directly in `QueryResponse`, not just logged.
- Weak matches are filtered by a minimum similarity threshold
  (`min_similarity_score` in `config.py`) before they ever reach the LLM.
