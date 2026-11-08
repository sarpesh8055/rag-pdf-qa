"""STEP 2 - CHUNK: split long pages into small, overlapping pieces.

Why? An embedding of a whole page is "blurry" - it mixes many topics.
Small chunks give sharper embeddings, so retrieval finds the exact
passage that answers the question. The overlap stops a sentence from
being cut in half at a chunk boundary and losing its meaning.
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraphs first, then lines, then sentences, then words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)  # metadata (source, page) is copied to each chunk
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
