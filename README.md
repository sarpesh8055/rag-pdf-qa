# 📄 RAG-Based Document Q&A Assistant

Upload PDFs and ask questions about them. Answers come **only** from your documents and cite the file and page they came from.

**Stack:** Python · LangChain · FAISS · Hugging Face (sentence-transformers) · Groq / Ollama · Streamlit

---

## How RAG works (the 30-second version)

An LLM doesn't know what's in *your* PDFs. RAG (Retrieval-Augmented Generation) fixes that by **looking up the relevant passages first** and handing them to the LLM with the question.

```
            INDEXING (once)                                 QUERYING (every question)
┌────────┐   ┌────────┐   ┌──────────┐   ┌───────┐     ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌─────┐
│  PDFs  │──▶│  Load  │──▶│  Chunk   │──▶│ Embed │──┐  │ Question │──▶│  Embed   │──▶│ FAISS top-k  │──▶│ LLM │──▶ Answer + [1][2]
└────────┘   │ (page) │   │ 800 char │   │ +FAISS│  │  └──────────┘   └──────────┘   │ search       │   └─────┘
             └────────┘   └──────────┘   └───────┘  └──────────────────────────────▶└──────────────┘
```

| Step | File | What it does |
|---|---|---|
| 1. Load | `rag/loader.py` | PDF → one text document per page, keeping `source` + `page` metadata |
| 2. Chunk | `rag/chunker.py` | Splits pages into ~800-char overlapping chunks |
| 3. Embed + index | `rag/vectorstore.py` | Converts chunks to 384-dim vectors (all-MiniLM-L6-v2) and stores them in FAISS |
| 4. Retrieve | `rag/pipeline.py` | Embeds the question, finds the k nearest chunks |
| 5. Generate | `rag/pipeline.py` | Prompts the LLM with numbered chunks, asks it to answer only from them and cite `[n]` |

---

## Setup (≈10 minutes)

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the LLM
cp .env.example .env               # Windows: copy .env.example .env
```

Then pick **one** LLM option in `.env`:

- **Groq (easiest, free):** sign up at https://console.groq.com/keys, paste the key into `GROQ_API_KEY`, keep `LLM_PROVIDER=groq`.
- **Ollama (fully local/offline):** install from https://ollama.com, run `ollama pull llama3.2`, set `LLM_PROVIDER=ollama`.

## Run it

**Web app:**
```bash
streamlit run app.py
```
Upload PDFs in the sidebar → **Build index** → chat.

**Terminal:**
```bash
# put PDFs in data/pdfs/ first
python ingest.py
python ask.py "What is the main conclusion of the paper?"
```

> Good test documents: a lecture script from your Paderborn courses, a research paper from arXiv, or your own resume.

---

## Learn by experimenting

Change one thing at a time in `rag/config.py` and see what happens. This is exactly what interviewers like to hear about.

| Try | What you'll learn |
|---|---|
| `CHUNK_SIZE` = 200 vs 800 vs 2000 | Small chunks = precise but lose context; big chunks = context but noisy retrieval |
| `CHUNK_OVERLAP` = 0 | Answers that span a chunk boundary start failing |
| `TOP_K` = 1 vs 8 | Too few → missing info; too many → LLM gets distracted, slower |
| Ask something **not** in the PDF | Does it say "I couldn't find that"? (tests hallucination control) |
| `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` | Compare retrieval quality of different embedders |
| Look at the `distance` in Sources | Learn what a "good match" distance looks like for your data |

---

## Roadmap (make it stand out)

Do these in order; each one is a resume-worthy improvement.

1. **Evaluation set** – write 15–20 question/expected-answer pairs for one PDF, measure *retrieval hit rate* (is the right page in top-k?) and answer correctness. Report numbers.
2. **Hybrid search** – combine BM25 keyword search with FAISS (`EnsembleRetriever`). Helps with names, codes, formulas.
3. **Re-ranking** – retrieve 20 chunks, re-rank with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), keep top 4.
4. **Chat memory** – rewrite follow-up questions ("what about the second one?") into standalone questions before retrieval.
5. **Persistence** – save the Streamlit index to disk so it survives restarts.
6. **Deploy** – Streamlit Community Cloud (free) and put the link in your resume.

---

## Project structure

```
rag-pdf-qa/
├── app.py              # Streamlit UI
├── ingest.py           # CLI: build index from data/pdfs/
├── ask.py              # CLI: ask questions
├── rag/
│   ├── config.py       # all tunable settings
│   ├── loader.py       # step 1: PDF → pages
│   ├── chunker.py      # step 2: pages → chunks
│   ├── vectorstore.py  # step 3: embeddings + FAISS
│   ├── llm.py          # Groq / Ollama
│   └── pipeline.py     # steps 4-5: retrieve + generate with citations
├── data/pdfs/          # your PDFs (git-ignored)
├── requirements.txt
└── .env.example
```

## Troubleshooting

- **"No extractable text"** – the PDF is a scan (images). Needs OCR (e.g. `pytesseract`), a good roadmap item.
- **Groq 401 error** – API key missing or wrong in `.env`.
- **Ollama connection refused** – start the Ollama app / run `ollama serve`.
- **First run is slow** – the embedding model (~90 MB) downloads once, then it's cached.
