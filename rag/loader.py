"""STEP 1 - LOAD: turn PDFs into text, one Document per page.

We keep the file name and page number as metadata so that every
answer can later cite exactly where its information came from.
"""
from pathlib import Path
from typing import IO, List, Union

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file: Union[str, Path, IO[bytes]], source_name: str = None) -> List[Document]:
    """Read one PDF (path or file-like object) and return a list of page Documents."""
    reader = PdfReader(file)
    name = source_name or Path(str(file)).name

    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:  # skip empty / image-only pages
            continue
        pages.append(Document(page_content=text, metadata={"source": name, "page": i + 1}))
    return pages


def load_folder(folder: Union[str, Path]) -> List[Document]:
    """Load every PDF inside a folder."""
    docs = []
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        docs.extend(load_pdf(pdf_path))
    return docs
