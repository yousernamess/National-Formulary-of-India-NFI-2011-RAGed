# National Formulary of India (NFI 2011) — RAG System

AI-powered drug information assistant that lets you ask natural-language questions
about the **National Formulary of India 2011** (800 pages) and receive precise,
cited answers using Retrieval-Augmented Generation (RAG).

---

## Architecture

```
NFI_2011.pdf
     │
     ▼
ingest.py          ← pypdf + LangChain text splitter
     │  chunks + embeddings (all-MiniLM-L6-v2, runs locally)
     ▼
chroma_db/         ← ChromaDB persistent vector store
     │
     ▼
rag.py             ← retriever + OpenAI LLM (gpt-4o-mini)
     │
     ▼
app.py             ← Streamlit web UI
```

| Component | Library / Model |
|-----------|----------------|
| PDF parsing | `pypdf` |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | ChromaDB (persistent on disk) |
| LLM | OpenAI `gpt-4o-mini` (configurable) |
| UI | Streamlit |

---

## Quick Start

### 1 — Clone and install dependencies

```bash
git clone https://github.com/yousernamess/National-Formulary-of-India-NFI-2011-RAGed.git
cd National-Formulary-of-India-NFI-2011-RAGed

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Obtain the NFI 2011 PDF

Download the official PDF from the Ministry of Health and Family Welfare, Government
of India and save it locally (e.g. `NFI_2011.pdf`).

### 3 — Ingest the PDF

```bash
python ingest.py --pdf NFI_2011.pdf
```

This will:
- Extract text from all 800 pages
- Split text into ~800-token chunks (with 100-token overlap)
- Embed each chunk with `all-MiniLM-L6-v2` (runs on CPU, no API key needed)
- Store everything in `chroma_db/` (created automatically)

Ingestion takes ~5–15 minutes on a CPU for an 800-page PDF.

**Optional flags:**

```bash
python ingest.py --pdf NFI_2011.pdf \
    --chroma-dir my_db \      # custom vector store path
    --collection  nfi_2011    # custom collection name
```

### 4 — Set your OpenAI API key

```bash
cp .env.example .env          # then edit .env
# OR export directly:
export OPENAI_API_KEY="sk-..."
```

### 5a — Launch the web UI (recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5b — CLI usage

```bash
python rag.py "What are the contraindications of Metformin?"
python rag.py "List the active ingredients in ORS."
python rag.py "What is the adult dose of Amoxicillin?"
```

---

## Configuration

All settings can be overridden with environment variables (or placed in a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `CHROMA_DIR` | `chroma_db` | Path to ChromaDB persistent store |
| `COLLECTION_NAME` | `nfi_2011` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk (ingest only) |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks (ingest only) |
| `TOP_K` | `5` | Number of chunks retrieved per query |

---

## .env.example

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CHROMA_DIR=chroma_db
COLLECTION_NAME=nfi_2011
TOP_K=5
```

---

## File Structure

```
.
├── ingest.py          # PDF → ChromaDB ingestion pipeline
├── rag.py             # RAG query pipeline (also usable as CLI)
├── app.py             # Streamlit web UI
├── requirements.txt   # Python dependencies
├── .env.example       # Example environment variables
├── .gitignore
└── README.md
```

---

## Notes

- The `chroma_db/` directory and `*.pdf` files are excluded from version control (see
  `.gitignore`). Only code is committed.
- By default the system uses `gpt-4o-mini` which is cost-efficient. You can switch to
  `gpt-4o` for higher quality answers by setting `OPENAI_MODEL=gpt-4o`.
- To use a different LLM provider, replace `ChatOpenAI` in `rag.py` with any
  LangChain-compatible chat model (e.g. `ChatAnthropic`, `ChatGoogleGenerativeAI`).
