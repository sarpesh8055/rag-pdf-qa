"""Build the vector index from every PDF in data/pdfs/.

Usage:  python ingest.py
"""
from rag.chunker import split_documents
from rag.config import PDF_DIR, VECTORSTORE_DIR
from rag.loader import load_folder
from rag.vectorstore import build_index, save_index


def main():
    pages = load_folder(PDF_DIR)
    if not pages:
        print(f"No PDFs with text found in {PDF_DIR}/ - add some and re-run.")
        return
    print(f"Loaded {len(pages)} pages")

    chunks = split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding chunks (first run downloads the embedding model)...")
    index = build_index(chunks)
    save_index(index)
    print(f"Saved FAISS index to {VECTORSTORE_DIR}/")


if __name__ == "__main__":
    main()
