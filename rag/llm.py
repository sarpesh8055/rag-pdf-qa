"""The LLM that writes the final answer. Swap providers with LLM_PROVIDER in .env."""
from rag.config import GROQ_MODEL, LLM_PROVIDER, OLLAMA_MODEL


def get_llm(temperature: float = 0.0):
    # temperature=0 -> deterministic, factual answers (what we want for Q&A)
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, temperature=temperature)
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'ollama'.")
