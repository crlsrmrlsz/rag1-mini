"""RAG1-Mini Search Interface.

A Streamlit application for testing the RAG system with Weaviate backend.
Now includes query preprocessing (classification, step-back prompting) and
LLM-based answer generation.

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

from src.config import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    AVAILABLE_GENERATION_MODELS,
    GENERATION_MODEL,
    ENABLE_ANSWER_GENERATION,
    ENABLE_QUERY_PREPROCESSING,
)
from src.ui.services.search import search_chunks, list_collections
from src.preprocessing import preprocess_query, QueryType
from src.generation import generate_answer


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

if "preprocessed_query" not in st.session_state:
    st.session_state.preprocessed_query = None

if "generated_answer" not in st.session_state:
    st.session_state.generated_answer = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _display_chunks(chunks):
    """Display chunk results with expandable details."""
    for i, chunk in enumerate(chunks, 1):
        # Extract author for cleaner display
        book_parts = chunk["book_id"].rsplit("(", 1)
        book_title = book_parts[0].strip()
        author = book_parts[1].rstrip(")") if len(book_parts) > 1 else ""

        with st.expander(
            f"**[{i}]** {book_title[:50]}... | Score: {chunk['similarity']:.3f}",
            expanded=(i <= 3 and not st.session_state.generated_answer),
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

# Number of results
top_k = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=MAX_TOP_K,
    value=DEFAULT_TOP_K,
    help="How many chunks to retrieve.",
)

# -----------------------------------------------------------------------------
# Answer Generation Configuration
# -----------------------------------------------------------------------------
st.sidebar.subheader("Answer Generation")

# Enable answer generation
enable_generation = st.sidebar.checkbox(
    "Generate Answer",
    value=ENABLE_ANSWER_GENERATION,
    help="Use an LLM to synthesize an answer from retrieved chunks.",
)

# Enable query preprocessing
enable_preprocessing = st.sidebar.checkbox(
    "Query Preprocessing",
    value=ENABLE_QUERY_PREPROCESSING,
    help="Classify queries and apply step-back prompting for open-ended questions.",
)

# Model selection for generation
if enable_generation:
    model_options = {model_id: label for model_id, label in AVAILABLE_GENERATION_MODELS}
    selected_model = st.sidebar.selectbox(
        "Generation Model",
        options=list(model_options.keys()),
        index=2,  # Default to gpt-5-mini (index 2)
        format_func=lambda x: model_options[x],
        help="Model used for answer generation. Higher cost = better quality.",
    )
else:
    selected_model = GENERATION_MODEL

# Reranking toggle (disabled by default - too slow on CPU)
with st.sidebar.expander("Advanced Options", expanded=False):
    use_reranking = st.sidebar.checkbox(
        "Enable Cross-Encoder Reranking",
        value=False,
        help="Re-scores results with a cross-encoder for higher accuracy. "
             "Very slow on CPU (~2 min/query). Disabled by default.",
    )

# Show current configuration summary
with st.sidebar.expander("Current Configuration", expanded=False):
    config_summary = f"""
**Search Type:** {search_type}
**Alpha:** {alpha if search_type == 'hybrid' else 'N/A'}
**Top-K:** {top_k}
**Collection:** {selected_collection if selected_collection else 'None'}
**Preprocessing:** {'Enabled' if enable_preprocessing else 'Disabled'}
**Generation:** {'Enabled' if enable_generation else 'Disabled'}
**Model:** {selected_model if enable_generation else 'N/A'}
**Reranking:** {'Enabled' if use_reranking else 'Disabled'}
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
        # Step 1: Query Preprocessing (optional)
        preprocessed = None
        search_query = query

        if enable_preprocessing:
            with st.spinner("Analyzing query..."):
                try:
                    preprocessed = preprocess_query(query)
                    search_query = preprocessed.search_query
                    st.session_state.preprocessed_query = preprocessed
                except Exception as e:
                    st.warning(f"Preprocessing failed: {e}. Using original query.")
                    st.session_state.preprocessed_query = None
        else:
            st.session_state.preprocessed_query = None

        # Step 2: Search
        with st.spinner("Searching..."):
            try:
                results = search_chunks(
                    query=search_query,
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
                st.session_state.generated_answer = None
                preprocessed = None

        # Step 3: Answer Generation (optional)
        if enable_generation and st.session_state.search_results:
            with st.spinner("Generating answer..."):
                try:
                    query_type = preprocessed.query_type if preprocessed else QueryType.FACTUAL
                    answer = generate_answer(
                        query=query,
                        chunks=st.session_state.search_results,
                        query_type=query_type,
                        model=selected_model,
                    )
                    st.session_state.generated_answer = answer
                except Exception as e:
                    st.warning(f"Answer generation failed: {e}")
                    st.session_state.generated_answer = None
        else:
            st.session_state.generated_answer = None


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if st.session_state.search_results:
    st.divider()
    st.subheader(f"Results for: \"{st.session_state.last_query}\"")

    # -------------------------------------------------------------------------
    # Query Analysis Section (if preprocessing was enabled)
    # -------------------------------------------------------------------------
    if st.session_state.preprocessed_query:
        prep = st.session_state.preprocessed_query
        with st.expander("Query Analysis", expanded=True):
            col1, col2 = st.columns(2)

            # Query type with color-coded badge
            type_colors = {
                QueryType.FACTUAL: "blue",
                QueryType.OPEN_ENDED: "green",
                QueryType.MULTI_HOP: "orange",
            }
            type_color = type_colors.get(prep.query_type, "gray")
            col1.markdown(f"**Type:** :{type_color}[{prep.query_type.value.upper()}]")
            col2.markdown(f"**Preprocessing Time:** {prep.preprocessing_time_ms:.0f}ms")

            # Show step-back query if applied
            if prep.step_back_query and prep.step_back_query != prep.original_query:
                st.markdown("**Step-Back Query:**")
                st.info(prep.step_back_query)

    # -------------------------------------------------------------------------
    # Generated Answer Section (if generation was enabled)
    # -------------------------------------------------------------------------
    if st.session_state.generated_answer:
        ans = st.session_state.generated_answer
        st.markdown("### Generated Answer")

        # Display the answer
        st.markdown(ans.answer)

        # Show metadata
        col1, col2, col3 = st.columns(3)
        col1.caption(f"Model: {ans.model}")
        col2.caption(f"Sources cited: {ans.sources_used}")
        col3.caption(f"Generated in {ans.generation_time_ms:.0f}ms")

        st.divider()

    # -------------------------------------------------------------------------
    # Retrieved Chunks Section
    # -------------------------------------------------------------------------
    chunks_header = f"Retrieved Chunks ({len(st.session_state.search_results)})"

    # Make chunks collapsible if answer was generated (focus on answer)
    if st.session_state.generated_answer:
        with st.expander(chunks_header, expanded=False):
            _display_chunks(st.session_state.search_results)
    else:
        st.markdown(f"### {chunks_header}")
        _display_chunks(st.session_state.search_results)

elif query and not st.session_state.search_results:
    st.info("No results found. Try a different query.")

else:
    st.info("Enter a query above to search the knowledge base.")


# ============================================================================
# FOOTER - Educational Notes
# ============================================================================

with st.expander("How This Works"):
    st.markdown("""
    ### Complete RAG Pipeline Flow

    1. **Query Preprocessing** (optional): Classifies your query and applies
       transformations for better retrieval:
       - FACTUAL queries use direct search
       - OPEN_ENDED queries get step-back prompting (broader concepts)
       - MULTI_HOP queries (coming soon) will be decomposed

    2. **Vector Search**: Your query is converted to an embedding and matched
       against document chunks using cosine similarity.

    3. **Hybrid Search** (default): Combines vector similarity with BM25
       keyword matching for better term-specific retrieval.

    4. **Answer Generation** (optional): An LLM synthesizes a coherent answer
       from the retrieved chunks, with source citations.

    ### Query Preprocessing

    **Step-Back Prompting** (for open-ended questions):

    Research shows that abstracting queries to broader concepts improves
    retrieval for philosophical and wisdom-seeking questions.

    Example:
    - Original: "How should I live my life?"
    - Step-back: "Stoic and philosophical principles for living a good life"

    The broader query retrieves more diverse, relevant passages.

    ### Search Types

    - **Semantic (Vector)**: Finds chunks with similar *meaning*, even if
      they use different words. Good for concepts and ideas.

    - **Hybrid**: Combines semantic search with keyword matching (BM25).
      Good for technical terms that should match exactly.

    ### Answer Generation

    After retrieving relevant chunks, an LLM synthesizes a coherent answer:
    - Uses query-type-specific prompts (factual vs philosophical)
    - Cites sources by number [1], [2], etc.
    - Configurable model selection (budget to premium)

    ### Technical Details

    - **Embedding Model**: text-embedding-3-large (3072 dimensions)
    - **Vector Database**: Weaviate with HNSW index
    - **Distance Metric**: Cosine similarity (1.0 = identical)
    - **Chunk Size**: ~800 tokens with 2-sentence overlap
    - **Generation Models**: GPT-5 Nano/Mini, DeepSeek, Gemini Flash, Claude Haiku

    ### Evaluation Results (RAGAS)

    | Configuration | Relevancy | Faithfulness |
    |---------------|-----------|--------------|
    | Vector, top_k=5 | 0.67 | 0.93 |
    | Hybrid, top_k=10 | 0.79 | 0.89 |
    | Hybrid + Rerank | 0.79 | 0.93 |
    """)
