"""STEP 4 + 5 - RETRIEVE + GENERATE: the actual RAG loop.

1. Embed the question and fetch the TOP_K most similar chunks from FAISS.
2. Paste those chunks into a prompt, each labelled [1], [2], ...
3. Ask the LLM to answer ONLY from that context and cite the labels.
"""
from typing import Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.config import TOP_K
from rag.llm import get_llm

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's documents.

Rules:
- Use ONLY the information in the context below. Do not use outside knowledge.
- After each fact, cite the source number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, say: "I couldn't find that in the documents."
- Be concise.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def retrieve(index: FAISS, question: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    """Return (chunk, distance) pairs. Lower distance = more similar."""
    return index.similarity_search_with_score(question, k=k)


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        blocks.append(f"[{i}] (source: {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}")
    return "\n\n".join(blocks)


def answer(index: FAISS, question: str, k: int = TOP_K) -> Dict:
    hits = retrieve(index, question, k)
    docs = [d for d, _ in hits]

    chain = PROMPT | get_llm() | StrOutputParser()
    reply = chain.invoke({"context": format_context(docs), "question": question})

    sources = [
        {
            "n": i,
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "score": round(float(score), 3),
            "text": d.page_content,
        }
        for i, (d, score) in enumerate(hits, start=1)
    ]
    return {"answer": reply, "sources": sources}
