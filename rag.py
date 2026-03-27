"""
rag.py
------
Core Retrieval-Augmented Generation (RAG) pipeline for the
National Formulary of India (NFI) 2011.

The module exposes a single public function ``answer(question)`` that:
  1. Retrieves the top-k most relevant chunks from the ChromaDB vector store.
  2. Builds a prompt with those chunks as context.
  3. Calls the configured LLM to produce a grounded answer.

Environment variables
---------------------
OPENAI_API_KEY   – Required for the default OpenAI backend.
OPENAI_MODEL     – Model name (default: gpt-4o-mini).
CHROMA_DIR       – Path to the ChromaDB store (default: chroma_db).
COLLECTION_NAME  – Collection name (default: nfi_2011).
EMBEDDING_MODEL  – HuggingFace model for query embedding
                   (default: sentence-transformers/all-MiniLM-L6-v2).
TOP_K            – Number of chunks to retrieve (default: 5).
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nfi_2011")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a helpful medical information assistant specialised in \
the National Formulary of India (NFI) 2011. Answer the user's question using ONLY \
the context passages provided below. If the answer cannot be found in the context, \
say "I could not find relevant information in the NFI 2011 for your query."

Context:
{context}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Cached components (initialised once per process)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_retriever():
    """Load the ChromaDB retriever (cached after first call)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


@lru_cache(maxsize=1)
def _get_llm():
    """Instantiate the LLM (cached after first call)."""
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _format_docs(docs) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | NFI p.{page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer(question: str) -> dict:
    """
    Run the RAG pipeline for *question* and return a dict with keys:

    - ``answer``  : the LLM-generated answer string
    - ``sources`` : list of ``{page, chunk, text}`` dicts for each retrieved chunk
    """
    retriever = _get_retriever()
    llm = _get_llm()

    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Build context string
    context = _format_docs(docs)

    # Call the LLM
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    response = chain.invoke({"context": context, "question": question})

    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "chunk": doc.metadata.get("chunk", "?"),
            "text": doc.page_content,
        }
        for doc in docs
    ]

    return {"answer": response, "sources": sources}


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag.py \"<your question>\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = answer(question)
    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for src in result["sources"]:
        print(f"  • NFI page {src['page']} (chunk {src['chunk']}): "
              f"{src['text'][:120].replace('\\n', ' ')} …")
