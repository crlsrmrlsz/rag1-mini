"""RAG1-Mini Search Interface.

A Streamlit application for testing the RAG system with Weaviate backend.
Features:
- Query preprocessing (classification, step-back prompting)
- Hybrid/vector search with optional cross-encoder reranking
- LLM-based answer generation
- Pipeline logging with full prompt visibility

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
import pandas as pd

from src.config import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    AVAILABLE_GENERATION_MODELS,
    AVAILABLE_PREPROCESSING_MODELS,
    GENERATION_MODEL,
    PREPROCESSING_MODEL,
    ENABLE_ANSWER_GENERATION,
    ENABLE_QUERY_PREPROCESSING,
)
from src.ui.services.search import search_chunks, list_collections
from src.preprocessing import preprocess_query, QueryType
from src.generation import generate_answer
from src.utils.openrouter_models import (
    fetch_available_models,
    get_preprocessing_models,
    get_generation_models,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG1-Mini Search",
    page_icon="books",
    layout="wide",
)


# ============================================================================
# DYNAMIC MODEL LOADING
# ============================================================================
# Fetch models from OpenRouter API, cached for 1 hour to avoid repeated calls


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_models():
    """Fetch and cache available models from OpenRouter API.

    Returns tuple of (preprocessing_models, generation_models).
    Falls back to config defaults if API fails.
    """
    models = fetch_available_models()
    if models:
        prep_models = get_preprocessing_models(models)
        gen_models = get_generation_models(models)
        return prep_models, gen_models
    # Fallback to config defaults
    return AVAILABLE_PREPROCESSING_MODELS, AVAILABLE_GENERATION_MODELS


# Load models (cached)
DYNAMIC_PREPROCESSING_MODELS, DYNAMIC_GENERATION_MODELS = get_cached_models()


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

if "rerank_data" not in st.session_state:
    st.session_state.rerank_data = None

if "retrieval_settings" not in st.session_state:
    st.session_state.retrieval_settings = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _display_chunks(chunks, show_indices=True):
    """Display chunk results with expandable details."""
    for i, chunk in enumerate(chunks, 1):
        # Extract author for cleaner display
        book_parts = chunk["book_id"].rsplit("(", 1)
        book_title = book_parts[0].strip()
        author = book_parts[1].rstrip(")") if len(book_parts) > 1 else ""

        prefix = f"[{i}] " if show_indices else ""
        with st.expander(
            f"**{prefix}**{book_title[:50]}... | Score: {chunk['similarity']:.3f}",
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


def _render_pipeline_log():
    """Render the Pipeline Log tab with detailed prompts and transformations."""
    prep = st.session_state.preprocessed_query
    ans = st.session_state.generated_answer
    rerank = st.session_state.rerank_data
    settings = st.session_state.retrieval_settings

    # Stage 1: Query Preprocessing
    with st.expander("Stage 1: Query Preprocessing", expanded=True):
        if prep:
            col1, col2 = st.columns(2)

            # Query type with color-coded badge
            type_colors = {
                QueryType.FACTUAL: "blue",
                QueryType.OPEN_ENDED: "green",
                QueryType.MULTI_HOP: "orange",
            }
            type_color = type_colors.get(prep.query_type, "gray")
            col1.markdown(f"**Query Type:** :{type_color}[{prep.query_type.value.upper()}]")
            col2.metric("Time", f"{prep.preprocessing_time_ms:.0f}ms")

            # Show complete classification prompt (system + user)
            st.markdown("**Classification Prompt (sent to LLM):**")
            classification_full = f"[System]\n{prep.classification_prompt_used}\n\n[User]\n{prep.original_query}"
            st.code(classification_full, language="text")

            # Show classification LLM response
            if prep.classification_response:
                st.markdown("**LLM Response:**")
                st.code(prep.classification_response, language="json")

            if prep.step_back_query and prep.step_back_query != prep.original_query:
                st.divider()

                # Show complete step-back prompt (system + user)
                st.markdown("**Step-Back Prompt (sent to LLM):**")
                step_back_full = f"[System]\n{prep.step_back_prompt_used}\n\n[User]\n{prep.original_query}"
                st.code(step_back_full, language="text")

                # Show step-back LLM response
                if prep.step_back_response:
                    st.markdown("**LLM Response (Step-Back Query):**")
                    st.code(prep.step_back_response, language="text")

                st.markdown("**Final Search Query:**")
                st.info(prep.step_back_query)
        else:
            st.info("Preprocessing was disabled for this query.")

    # Stage 2: Retrieval
    with st.expander("Stage 2: Retrieval", expanded=True):
        if settings:
            col1, col2, col3 = st.columns(3)
            col1.metric("Search Type", settings.get("search_type", "N/A"))
            col2.metric("Alpha", settings.get("alpha", "N/A"))
            col3.metric("Top-K", settings.get("top_k", "N/A"))

            search_q = prep.search_query if prep else st.session_state.last_query
            st.markdown(f"**Query Sent to Weaviate:**")
            st.code(search_q, language="text")

            st.metric("Results Retrieved", len(st.session_state.search_results))
        else:
            st.info("No retrieval data available.")

    # Stage 3: Reranking
    with st.expander("Stage 3: Reranking", expanded=True):
        if rerank:
            col1, col2 = st.columns(2)
            col1.markdown(f"**Model:** `{rerank.model}`")
            col2.metric("Time", f"{rerank.rerank_time_ms:.0f}ms")

            if rerank.order_changes:
                st.markdown("**Order Changes (how rankings shifted):**")

                # Create a DataFrame for display
                df = pd.DataFrame(rerank.order_changes)
                df = df[["before_rank", "after_rank", "before_score", "after_score", "text_preview"]]
                df.columns = ["Before Rank", "After Rank", "Before Score", "After Score", "Text Preview"]
                df["Before Score"] = df["Before Score"].round(3)
                df["After Score"] = df["After Score"].round(3)

                st.dataframe(df, use_container_width=True)
        else:
            st.info("Reranking was disabled for this query.")

    # Stage 4: Answer Generation
    with st.expander("Stage 4: Answer Generation", expanded=True):
        if ans:
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Model:** `{ans.model}`")
            if ans.query_type:
                col2.markdown(f"**Query Type:** `{ans.query_type.value}`")
            col3.metric("Time", f"{ans.generation_time_ms:.0f}ms")

            # Show complete generation prompt (system + user)
            st.markdown("**Generation Prompt (sent to LLM):**")
            generation_full = f"[System]\n{ans.system_prompt_used}\n\n[User]\n{ans.user_prompt_used}"
            st.code(generation_full, language="text")

            # Show LLM response (the generated answer)
            st.markdown("**LLM Response:**")
            st.code(ans.answer, language="text")

            st.markdown(f"**Sources Cited:** {ans.sources_used}")
        else:
            st.info("Answer generation was disabled for this query.")


# ============================================================================
# SIDEBAR - Settings organized by Pipeline Stage
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
        "Collection",
        options=available_collections,
        help="Different collections may use different embedding models or chunking strategies.",
    )
else:
    st.sidebar.warning("No collections found. Is Weaviate running?")
    selected_collection = None

st.sidebar.divider()

# -----------------------------------------------------------------------------
# STAGE 1: Query Preprocessing
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Stage 1: Query Preprocessing")

enable_preprocessing = st.sidebar.checkbox(
    "Enable Preprocessing",
    value=ENABLE_QUERY_PREPROCESSING,
    help="Classify queries and apply step-back prompting for open-ended questions.",
)

if enable_preprocessing:
    prep_model_options = {model_id: label for model_id, label in DYNAMIC_PREPROCESSING_MODELS}
    selected_prep_model = st.sidebar.selectbox(
        "Preprocessing Model",
        options=list(prep_model_options.keys()),
        index=0,  # Default to first (cheapest)
        format_func=lambda x: prep_model_options[x],
        help="Model used for query classification and step-back prompting. (Fetched from OpenRouter)",
    )
else:
    selected_prep_model = PREPROCESSING_MODEL

st.sidebar.divider()

# -----------------------------------------------------------------------------
# STAGE 2: Retrieval
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Stage 2: Retrieval")

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

st.sidebar.divider()

# -----------------------------------------------------------------------------
# STAGE 3: Reranking
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Stage 3: Reranking")

use_reranking = st.sidebar.checkbox(
    "Enable Cross-Encoder Reranking",
    value=False,
    help="Re-scores results with a cross-encoder for higher accuracy.",
)

if use_reranking:
    st.sidebar.caption("Slow on CPU (~2 min/query). Retrieves 50 candidates, reranks to top-k.")

st.sidebar.divider()

# -----------------------------------------------------------------------------
# STAGE 4: Answer Generation
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Stage 4: Answer Generation")

enable_generation = st.sidebar.checkbox(
    "Enable Answer Generation",
    value=ENABLE_ANSWER_GENERATION,
    help="Use an LLM to synthesize an answer from retrieved chunks.",
)

if enable_generation:
    model_options = {model_id: label for model_id, label in DYNAMIC_GENERATION_MODELS}
    selected_model = st.sidebar.selectbox(
        "Generation Model",
        options=list(model_options.keys()),
        index=min(1, len(model_options) - 1),  # Default to second option (balanced)
        format_func=lambda x: model_options[x],
        help="Model used for answer generation. (Fetched from OpenRouter)",
    )
else:
    selected_model = GENERATION_MODEL

st.sidebar.divider()

# Show current configuration summary
with st.sidebar.expander("Current Configuration", expanded=False):
    config_summary = f"""
**Stage 1: Preprocessing**
- Enabled: {'Yes' if enable_preprocessing else 'No'}
- Model: {selected_prep_model if enable_preprocessing else 'N/A'}

**Stage 2: Retrieval**
- Type: {search_type}
- Alpha: {alpha if search_type == 'hybrid' else 'N/A'}
- Top-K: {top_k}
- Collection: {selected_collection if selected_collection else 'None'}

**Stage 3: Reranking**
- Enabled: {'Yes' if use_reranking else 'No'}

**Stage 4: Generation**
- Enabled: {'Yes' if enable_generation else 'No'}
- Model: {selected_model if enable_generation else 'N/A'}
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
            with st.spinner("Stage 1: Analyzing query..."):
                try:
                    preprocessed = preprocess_query(query, model=selected_prep_model)
                    search_query = preprocessed.search_query
                    st.session_state.preprocessed_query = preprocessed
                except Exception as e:
                    st.warning(f"Preprocessing failed: {e}. Using original query.")
                    st.session_state.preprocessed_query = None
        else:
            st.session_state.preprocessed_query = None

        # Step 2 & 3: Search (with optional reranking)
        with st.spinner("Stage 2: Searching..." if not use_reranking else "Stage 2-3: Searching and reranking..."):
            try:
                search_output = search_chunks(
                    query=search_query,
                    top_k=top_k,
                    search_type=search_type,
                    alpha=alpha,
                    collection_name=selected_collection,
                    use_reranking=use_reranking,
                )
                st.session_state.search_results = search_output.results
                st.session_state.rerank_data = search_output.rerank_data
                st.session_state.last_query = query
                st.session_state.retrieval_settings = {
                    "search_type": search_type,
                    "alpha": alpha,
                    "top_k": top_k,
                }
                st.session_state.connection_error = None
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = []
                st.session_state.generated_answer = None
                st.session_state.rerank_data = None
                preprocessed = None

        # Step 4: Answer Generation (optional)
        if enable_generation and st.session_state.search_results:
            with st.spinner("Stage 4: Generating answer..."):
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
# RESULTS DISPLAY - Tabs: Answer | Pipeline Log | Chunks
# ============================================================================

if st.session_state.search_results:
    st.divider()
    st.subheader(f"Results for: \"{st.session_state.last_query}\"")

    # Create tabs for different views
    tab_answer, tab_log, tab_chunks = st.tabs(["Answer", "Pipeline Log", "Retrieved Chunks"])

    # -------------------------------------------------------------------------
    # TAB 1: Answer
    # -------------------------------------------------------------------------
    with tab_answer:
        # Query Analysis Section (if preprocessing was enabled)
        if st.session_state.preprocessed_query:
            prep = st.session_state.preprocessed_query
            with st.container():
                st.markdown("#### Query Analysis")
                col1, col2, col3 = st.columns(3)

                # Query type with color-coded badge
                type_colors = {
                    QueryType.FACTUAL: "blue",
                    QueryType.OPEN_ENDED: "green",
                    QueryType.MULTI_HOP: "orange",
                }
                type_color = type_colors.get(prep.query_type, "gray")
                col1.markdown(f"**Type:** :{type_color}[{prep.query_type.value.upper()}]")
                col2.markdown(f"**Time:** {prep.preprocessing_time_ms:.0f}ms")

                # Show step-back query if applied
                if prep.step_back_query and prep.step_back_query != prep.original_query:
                    st.info(f"**Step-Back Query:** {prep.step_back_query}")

                st.divider()

        # Generated Answer Section (if generation was enabled)
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

        else:
            st.info("Answer generation is disabled. Enable it in the sidebar to see synthesized answers.")

    # -------------------------------------------------------------------------
    # TAB 2: Pipeline Log
    # -------------------------------------------------------------------------
    with tab_log:
        st.markdown("### Pipeline Execution Log")
        st.caption("Full visibility into what happened at each stage of the RAG pipeline.")
        _render_pipeline_log()

    # -------------------------------------------------------------------------
    # TAB 3: Retrieved Chunks
    # -------------------------------------------------------------------------
    with tab_chunks:
        st.markdown(f"### Retrieved Chunks ({len(st.session_state.search_results)})")
        _display_chunks(st.session_state.search_results)

elif query and not st.session_state.search_results:
    st.info("No results found. Try a different query.")

else:
    st.info("Enter a query above to search the knowledge base.")


