"""RAG1-Mini Search Interface.

A Streamlit application for testing the RAG system with Weaviate backend.

Run with:
    streamlit run src/ui/app.py

Prerequisites:
    - Weaviate must be running (docker compose up -d)
    - Stage 6 must have been run to populate the collection
"""

import sys
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress noisy HTTP logs from Weaviate client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import streamlit as st

from src.config import DEFAULT_TOP_K, MAX_TOP_K
from src.ui.services.search import search_chunks, list_collections


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG1-Mini Search",
    page_icon="books",
    layout="wide",
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Streamlit reruns the entire script on every interaction.
# Session state persists data across reruns.

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "connection_error" not in st.session_state:
    st.session_state.connection_error = None


# ============================================================================
# SIDEBAR - Settings and Filters
# ============================================================================

st.sidebar.title("Settings")

# Collection selector (for future multiple embedding strategies)
try:
    available_collections = list_collections()
    st.session_state.connection_error = None
except Exception as e:
    available_collections = []
    st.session_state.connection_error = str(e)

if available_collections:
    selected_collection = st.sidebar.selectbox(
        "Embedding Collection",
        options=available_collections,
        help="Different collections may use different embedding models or chunking strategies.",
    )
else:
    st.sidebar.warning("No collections found. Is Weaviate running?")
    selected_collection = None

# -----------------------------------------------------------------------------
# Retrieval Strategy Configuration
# -----------------------------------------------------------------------------
st.sidebar.subheader("Retrieval Strategy")

# Search type selector
search_type = st.sidebar.radio(
    "Search Type",
    options=["vector", "hybrid"],
    index=1,  # Default to hybrid (better for this corpus)
    format_func=lambda x: "Semantic (Vector)" if x == "vector" else "Hybrid (Vector + Keyword)",
    help="Semantic search finds similar meaning. Hybrid also matches exact keywords.",
)

# Alpha slider for hybrid search
if search_type == "hybrid":
    alpha = st.sidebar.slider(
        "Hybrid Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0.0 = keyword only, 1.0 = vector only, 0.5 = balanced",
    )
else:
    alpha = 0.5

# Reranking toggle
use_reranking = st.sidebar.checkbox(
    "Enable Cross-Encoder Reranking",
    value=False,
    help="Uses a cross-encoder model to re-score results for higher accuracy. "
         "Slower but significantly improves result quality. "
         "First use downloads a 1.2GB model.",
)

# Number of results
top_k = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=MAX_TOP_K,
    value=DEFAULT_TOP_K,
    help="How many chunks to retrieve.",
)

# Show current configuration summary
with st.sidebar.expander("Current Configuration", expanded=False):
    config_summary = f"""
**Search Type:** {search_type}
**Alpha:** {alpha if search_type == 'hybrid' else 'N/A'}
**Reranking:** {'Enabled' if use_reranking else 'Disabled'}
**Top-K:** {top_k}
**Collection:** {selected_collection if selected_collection else 'None'}
    """
    st.markdown(config_summary.strip())


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("RAG1-Mini Search")
st.markdown("Search across 19 books combining neuroscience and philosophy.")

# Show connection error if any
if st.session_state.connection_error:
    st.error(f"Connection Error: {st.session_state.connection_error}")
    st.info("Make sure Weaviate is running: `docker compose up -d`")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the relationship between emotions and decision-making?",
)

# Search button
search_clicked = st.button("Search", type="primary", disabled=not query)

# Execute search
if search_clicked and query:
    if not selected_collection:
        st.error("No collection available. Please run `docker compose up -d` and run Stage 6.")
    else:
        with st.spinner("Searching..."):
            try:
                results = search_chunks(
                    query=query,
                    top_k=top_k,
                    search_type=search_type,
                    alpha=alpha,
                    collection_name=selected_collection,
                    use_reranking=use_reranking,
                )
                st.session_state.search_results = results
                st.session_state.last_query = query
                st.session_state.connection_error = None
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = []


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if st.session_state.search_results:
    st.divider()
    st.subheader(f"Results for: \"{st.session_state.last_query}\"")
    st.caption(f"Found {len(st.session_state.search_results)} chunks")

    for i, chunk in enumerate(st.session_state.search_results, 1):
        # Extract author for cleaner display
        book_parts = chunk["book_id"].rsplit("(", 1)
        book_title = book_parts[0].strip()
        author = book_parts[1].rstrip(")") if len(book_parts) > 1 else ""

        with st.expander(
            f"**{i}.** {book_title[:50]}... | Score: {chunk['similarity']:.3f}",
            expanded=(i <= 3),  # Expand top 3 by default
        ):
            # Metadata row
            col1, col2, col3 = st.columns(3)
            col1.metric("Similarity", f"{chunk['similarity']:.4f}")
            col2.metric("Tokens", chunk["token_count"])
            col3.markdown(f"**Author:** {author}")

            # Section info
            st.markdown(f"**Section:** {chunk['section']}")
            st.caption(f"Context: {chunk['context']}")

            # Main text
            st.markdown("---")
            st.markdown(chunk["text"])

elif query and not st.session_state.search_results:
    st.info("No results found. Try a different query.")

else:
    st.info("Enter a query above to search the knowledge base.")


# ============================================================================
# FOOTER - Educational Notes
# ============================================================================

with st.expander("How This Works"):
    st.markdown("""
    ### RAG Pipeline Flow

    1. **Your Query** is converted to a vector (embedding) using the same model
       that embedded all document chunks (text-embedding-3-large).

    2. **Vector Search** finds the chunks whose embeddings are most similar
       to your query embedding (cosine similarity).

    3. **Reranking (Optional)**: A cross-encoder re-scores the top-50 results
       by processing query and document together for deeper understanding.

    4. **Results** show the most relevant text passages with their source.

    ### Search Types

    - **Semantic (Vector)**: Finds chunks with similar *meaning*, even if
      they use different words. Good for concepts and ideas.

    - **Hybrid**: Combines semantic search with keyword matching (BM25).
      Good for technical terms that should match exactly.

    ### Cross-Encoder Reranking

    **Why Reranking Improves Results:**

    The default search uses a **bi-encoder** that embeds query and documents
    separately. This is fast but can miss subtle relationships.

    A **cross-encoder** processes query and document *together* through a
    transformer, enabling it to understand fine-grained semantic connections.

    Example:
    - Query: "What metaphor does Marcus Aurelius use for passions?"
    - Document: "He likens humans to puppets moved by wires"

    The bi-encoder might not connect "puppet metaphor" to "passions" because
    they're processed separately. The cross-encoder sees both together and
    understands that "puppets moved by wires" IS the metaphor about passions.

    **Trade-off:** Reranking is slower (~1-2s) but significantly more accurate.

    ### Technical Details

    - **Embedding Model**: text-embedding-3-large (3072 dimensions)
    - **Reranking Model**: mxbai-rerank-large-v1 (560M parameters)
    - **Vector Database**: Weaviate with HNSW index
    - **Distance Metric**: Cosine similarity (1.0 = identical)
    - **Chunk Size**: ~800 tokens with 2-sentence overlap

    ### Evaluation Metrics (RAGAS)

    - **Faithfulness**: Is the answer grounded in retrieved context?
    - **Answer Relevancy**: Does the answer address the question?
    - **Context Precision**: Are retrieved chunks relevant to the question?

    ### Configurations to Compare

    | Configuration | Relevancy | Notes |
    |---------------|-----------|-------|
    | Vector, top_k=5 | 0.67 | Baseline |
    | Hybrid, top_k=10 | 0.79 | +17% improvement |
    | Hybrid + Reranking | ??? | Expected +20-35% |
    """)
