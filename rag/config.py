"""Central settings. Everything tunable lives here so you can experiment easily."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# --- Chunking ---
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks

# --- Retrieval ---
TOP_K = 4              # how many chunks to send to the LLM

# --- Paths ---
VECTORSTORE_DIR = "vectorstore"
PDF_DIR = "data/pdfs"
