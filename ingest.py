"""
ingest.py
---------
Parse the National Formulary of India (NFI) PDF, split it into chunks,
embed each chunk with a local sentence-transformer model, and persist the
resulting vector store to disk (ChromaDB).

Usage
-----
    python ingest.py --pdf path/to/NFI_2011.pdf

The script is idempotent: running it again on the same PDF will overwrite
the existing collection.
"""

import argparse
import os
import sys

from tqdm import tqdm
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------------
# Configuration (override via environment variables or CLI flags)
# ---------------------------------------------------------------------------

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nfi_2011")

# Embedding model — runs entirely locally, no API key needed.
# "all-MiniLM-L6-v2" is fast and accurate for medical/drug text retrieval.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunk parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def extract_pages(pdf_path: str) -> list[dict]:
    """Return a list of ``{page_num, text}`` dicts from the PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(tqdm(reader.pages, desc="Extracting pages", unit="pg")):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page_num": i + 1, "text": text})
    return pages


def build_documents(pages: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Split page texts into chunks and return parallel lists of
    (chunk_texts, metadatas).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts: list[str] = []
    metadatas: list[dict] = []

    for page in tqdm(pages, desc="Chunking text", unit="pg"):
        chunks = splitter.split_text(page["text"])
        for chunk_idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": "NFI_2011",
                    "page": page["page_num"],
                    "chunk": chunk_idx,
                }
            )

    return texts, metadatas


def ingest(
    pdf_path: str,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB."""
    if not os.path.isfile(pdf_path):
        sys.exit(f"[ERROR] File not found: {pdf_path}")

    print(f"[INFO] Loading PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"[INFO] Extracted {len(pages)} non-empty pages.")

    texts, metadatas = build_documents(pages)
    print(f"[INFO] Created {len(texts)} text chunks.")

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"[INFO] Building ChromaDB collection '{collection_name}' at '{chroma_dir}' …")
    # Batch in groups of 500 to avoid memory spikes on large PDFs.
    batch_size = 500
    vectorstore = None
    for start in tqdm(range(0, len(texts), batch_size), desc="Indexing batches"):
        batch_texts = texts[start : start + batch_size]
        batch_meta = metadatas[start : start + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_meta,
                collection_name=collection_name,
                persist_directory=chroma_dir,
            )
        else:
            vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta)

    print(f"[INFO] Ingestion complete. Vector store saved to '{chroma_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest an NFI PDF into a ChromaDB vector store."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the National Formulary of India PDF file.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_DIR,
        help=f"Directory for the ChromaDB persistent store (default: {CHROMA_DIR}).",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args()

    ingest(args.pdf, chroma_dir=args.chroma_dir, collection_name=args.collection)
