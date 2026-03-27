"""
app.py
------
Streamlit web interface for the NFI 2011 RAG system.

Run:
    streamlit run app.py

Requirements:
    • Ingest the PDF first:  python ingest.py --pdf path/to/NFI_2011.pdf
    • Set OPENAI_API_KEY (or place it in a .env file).
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NFI 2011 — Drug Information Assistant",
    page_icon="💊",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💊 National Formulary of India (NFI) 2011")
st.subheader("AI-powered Drug Information Assistant")
st.markdown(
    """
    Ask any question about drugs, dosages, indications, contraindications, or
    ingredients listed in the **National Formulary of India 2011**.

    > **Before using**, ingest the PDF with:
    > ```bash
    > python ingest.py --pdf path/to/NFI_2011.pdf
    > ```
    > Then set your `OPENAI_API_KEY` in a `.env` file or as an environment variable.
    """
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API key. You can also set OPENAI_API_KEY in a .env file.",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    top_k = st.slider(
        "Chunks to retrieve (Top-K)",
        min_value=1,
        max_value=15,
        value=int(os.getenv("TOP_K", "5")),
        help="More chunks = more context, but slower and costlier.",
    )
    os.environ["TOP_K"] = str(top_k)

    st.markdown("---")
    st.caption("Vector store: `chroma_db/`")
    st.caption("Embedding: `all-MiniLM-L6-v2`")
    st.caption("LLM: `gpt-4o-mini` (configurable via OPENAI_MODEL)")

# ---------------------------------------------------------------------------
# Validate prerequisites before importing rag (to show friendly errors)
# ---------------------------------------------------------------------------

chroma_dir = os.getenv("CHROMA_DIR", "chroma_db")
if not os.path.isdir(chroma_dir):
    st.error(
        f"Vector store not found at `{chroma_dir}/`. "
        "Please run `python ingest.py --pdf path/to/NFI_2011.pdf` first."
    )
    st.stop()

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Please enter your OpenAI API key in the sidebar to enable answers.")

# ---------------------------------------------------------------------------
# Query form
# ---------------------------------------------------------------------------

with st.form("query_form"):
    question = st.text_area(
        "Your question",
        placeholder=(
            "e.g. What are the contraindications of Metformin?\n"
            "e.g. List the ingredients of ORS.\n"
            "e.g. What is the recommended dose of Amoxicillin for children?"
        ),
        height=120,
    )
    submitted = st.form_submit_button("🔍 Ask", use_container_width=True)

# ---------------------------------------------------------------------------
# Run RAG and display results
# ---------------------------------------------------------------------------

if submitted and question.strip():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("An OpenAI API key is required. Enter it in the sidebar.")
    else:
        with st.spinner("Retrieving relevant passages and generating answer …"):
            try:
                # Import here so that env vars (TOP_K, OPENAI_API_KEY) are set first.
                import rag as _rag

                # Invalidate caches so updated TOP_K takes effect immediately.
                _rag._get_retriever.cache_clear()
                _rag._get_llm.cache_clear()

                result = _rag.answer(question.strip())
            except Exception as exc:
                st.error(f"An error occurred: {exc}")
                st.stop()

        # --- Answer ---
        st.markdown("### 📋 Answer")
        st.markdown(result["answer"])

        # --- Sources ---
        if result["sources"]:
            st.markdown("### 📚 Source Passages")
            for i, src in enumerate(result["sources"], start=1):
                with st.expander(f"Passage {i} — NFI page {src['page']}"):
                    st.markdown(f"**Page:** {src['page']} | **Chunk:** {src['chunk']}")
                    st.text(src["text"])

elif submitted:
    st.warning("Please enter a question before clicking Ask.")
