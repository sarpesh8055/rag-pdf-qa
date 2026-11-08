"""Streamlit chat UI: upload PDFs, then chat with them.

Usage:  streamlit run app.py
"""
import streamlit as st

from rag.chunker import split_documents
from rag.config import LLM_PROVIDER, TOP_K
from rag.loader import load_pdf
from rag.pipeline import answer
from rag.vectorstore import build_index

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 Chat with your PDFs")
st.caption(f"Retrieval: FAISS + Hugging Face embeddings · LLM: {LLM_PROVIDER}")

# ---------- Sidebar: upload + index ----------
with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, TOP_K)

    if st.button("Build index", disabled=not files, use_container_width=True):
        with st.spinner("Reading, chunking and embedding..."):
            pages = []
            for f in files:
                pages.extend(load_pdf(f, source_name=f.name))
            if not pages:
                st.error("No extractable text found (scanned PDFs need OCR).")
            else:
                chunks = split_documents(pages)
                st.session_state.index = build_index(chunks)
                st.session_state.messages = []
                st.success(f"Indexed {len(pages)} pages → {len(chunks)} chunks")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['source']} — page {s['page']}** · distance `{s['score']}`")
            st.caption(s["text"][:500] + ("…" if len(s["text"]) > 500 else ""))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

if "index" not in st.session_state:
    st.info("Upload one or more PDFs in the sidebar and click **Build index** to start.")
elif question := st.chat_input("Ask something about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer(st.session_state.index, question, k=top_k)
            except Exception as e:
                st.error(f"LLM call failed: {e}\n\nCheck your .env (API key / Ollama running).")
                st.stop()
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
