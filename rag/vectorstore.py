"""STEP 3 - EMBED + INDEX: turn each chunk into a vector and store it in FAISS.

An embedding model maps text to a vector (here 384 numbers) so that texts
with similar meaning end up close together. FAISS is a library that finds
the nearest vectors to a query vector very quickly.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL, VECTORSTORE_DIR


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # Downloaded once from Hugging Face (~90 MB), then cached on disk.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )


def build_index(chunks: List[Document]) -> FAISS:
    return FAISS.from_documents(chunks, get_embeddings())


def save_index(index: FAISS, path: str = VECTORSTORE_DIR) -> None:
    index.save_local(path)


def load_index(path: str = VECTORSTORE_DIR) -> FAISS:
    if not Path(path).exists():
        raise FileNotFoundError(f"No index at '{path}'. Run `python ingest.py` first.")
    # FAISS stores metadata with pickle, so LangChain asks you to confirm you trust the file.
    # It's safe here because *you* created it.
    return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
