"""
Streamlit front-end for the RAG assistant.

Talks to the FastAPI backend over HTTP only — no business logic lives here,
keeping "app / API structure" cleanly separated from the "interface" layer.
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ShadowFox RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 ShadowFox RAG Assistant")
st.caption("Grounded Q&A over your PDF, TXT, and Markdown documents.")

if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar: ingestion + document list ---
with st.sidebar:
    st.header("📄 Documents")

    uploaded = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md", "markdown"])
    if uploaded is not None and st.button("Ingest document", use_container_width=True):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested '{data['filename']}' into {data['num_chunks']} chunks.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs_resp = requests.get(f"{API_URL}/documents", timeout=10)
        docs_resp.raise_for_status()
        docs_data = docs_resp.json()
        doc_options = {d["filename"]: d["doc_id"] for d in docs_data["documents"]}
        for d in docs_data["documents"]:
            st.text(f"• {d['filename']} ({d['num_chunks']} chunks)")
        st.caption(f"Total chunks indexed: {docs_data['total_chunks']}")
    except Exception as e:
        doc_options = {}
        st.warning(f"Backend not reachable yet: {e}")

    selected_names = st.multiselect(
        "Scope question to specific documents (optional)", options=list(doc_options.keys())
    )
    scoped_doc_ids = [doc_options[n] for n in selected_names] if selected_names else None

# --- Main: chat interface ---
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("citations"):
            with st.expander(f"Sources ({len(turn['citations'])})"):
                for c in turn["citations"]:
                    st.markdown(f"**{c['filename']}** — chunk {c['chunk_index']}")
                    st.caption(c["snippet"])
        g = turn.get("groundedness")
        if g:
            badge = "✅ grounded" if g["is_grounded"] else "⚠️ low confidence"
            st.caption(f"{badge} · confidence {g['confidence']:.2f} · {turn.get('latency_ms', 0)} ms")

question = st.chat_input("Ask a question about your documents...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking, and generating a grounded answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "doc_ids": scoped_doc_ids},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.write(data["answer"])
                if data.get("citations"):
                    with st.expander(f"Sources ({len(data['citations'])})"):
                        for c in data["citations"]:
                            st.markdown(f"**{c['filename']}** — chunk {c['chunk_index']}")
                            st.caption(c["snippet"])
                g = data.get("groundedness")
                if g:
                    badge = "✅ grounded" if g["is_grounded"] else "⚠️ low confidence"
                    st.caption(f"{badge} · confidence {g['confidence']:.2f} · {data['latency_ms']} ms")
                st.session_state.history.append(data)
            except Exception as e:
                st.error(f"Query failed: {e}")
