"""RAGLab Search Interface.

A Streamlit application for testing the RAG system with Weaviate backend.
Features:
- Query preprocessing (HyDE, decomposition strategies)
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
    AVAILABLE_PREPROCESSING_STRATEGIES,
    GENERATION_MODEL,
    PREPROCESSING_MODEL,
    get_valid_preprocessing_strategies,
)
from src.ui.services.search import search_chunks, get_available_collections, CollectionInfo
from src.rag_pipeline.retrieval.preprocessing import preprocess_query
from src.rag_pipeline.generation.answer_generator import generate_answer


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAGLab Search",
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

if "rerank_data" not in st.session_state:
    st.session_state.rerank_data = None

if "rrf_data" not in st.session_state:
    st.session_state.rrf_data = None

if "graph_metadata" not in st.session_state:
    st.session_state.graph_metadata = None

if "retrieval_settings" not in st.session_state:
    st.session_state.retrieval_settings = {}

# UI selection state (for progressive disclosure and reset logic)
if "ui_preprocessing_strategy" not in st.session_state:
    st.session_state.ui_preprocessing_strategy = "none"
if "ui_collection" not in st.session_state:
    st.session_state.ui_collection = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _find_section_collection(collection_infos: list[CollectionInfo]) -> str | None:
    """Find the section collection name from available collections."""
    for info in collection_infos:
        if info.strategy == "section":
            return info.collection_name
    return None




def _display_chunks(chunks, show_indices=True):
    """Display chunk results with expandable details."""
    for i, chunk in enumerate(chunks, 1):
        # Extract author for cleaner display
        book_parts = chunk["book_id"].rsplit("(", 1)
        book_title = book_parts[0].strip()
        author = book_parts[1].rstrip(")") if len(book_parts) > 1 else ""

        # RAPTOR summary indicator
        is_summary = chunk.get("is_summary", False)
        tree_level = chunk.get("tree_level", 0)
        summary_badge = " [SUMMARY]" if is_summary else ""

        prefix = f"[{i}] " if show_indices else ""
        with st.expander(
            f"**{prefix}**{book_title[:50]}...{summary_badge} | Score: {chunk['similarity']:.3f}",
            expanded=(i <= 3 and not st.session_state.generated_answer),
        ):
            # Metadata row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Similarity", f"{chunk['similarity']:.4f}")
            col2.metric("Tokens", chunk["token_count"])
            col3.markdown(f"**Author:** {author}")
            # Show tree level for RAPTOR chunks
            if tree_level > 0 or is_summary:
                col4.metric("Tree Level", tree_level)

            # Section info
            st.markdown(f"**Section:** {chunk['section']}")
            st.caption(f"Context: {chunk['context']}")

            # Main text
            st.markdown("---")
            st.markdown(chunk["text"])


def _render_preprocessing_stage(prep) -> None:
    """Render preprocessing stage details."""
    strategy = getattr(prep, 'strategy_used', 'N/A')
    model = getattr(prep, 'model', 'N/A')

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Strategy:** `{strategy}`")
    col2.markdown(f"**Model:** `{model}`")
    col3.metric("Time", f"{prep.preprocessing_time_ms:.0f}ms")

    # HyDE output
    hyde_passage = getattr(prep, 'hyde_passage', None)
    if hyde_passage and hyde_passage != prep.original_query:
        st.markdown("**Hypothetical Passage:**")
        st.info(hyde_passage)

    # Decomposition output
    sub_queries = getattr(prep, 'sub_queries', None)
    if sub_queries:
        st.markdown("**Sub-Questions:**")
        for i, sq in enumerate(sub_queries, 1):
            st.markdown(f"{i}. {sq}")


def _render_retrieval_stage(settings: dict, results: list, prep) -> None:
    """Render retrieval stage details."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Type", settings.get("search_type", "N/A"))
    col2.metric("Alpha", settings.get("alpha", "N/A"))
    col3.metric("Top-K", settings.get("top_k", "N/A"))
    col4.metric("Found", len(results))

    search_q = prep.search_query if prep else st.session_state.last_query
    st.markdown("**Search Query:**")
    st.code(search_q, language="text")


def _render_rrf_stage(rrf, prep) -> None:
    """Render RRF merging stage details."""
    col1, col2, col3 = st.columns(3)
    num_chunks = len(rrf.query_contributions) if hasattr(rrf, 'query_contributions') and rrf.query_contributions else 0
    num_queries = len(prep.generated_queries) if prep and hasattr(prep, 'generated_queries') else 0
    col1.metric("Queries Merged", num_queries)
    col2.metric("Unique Chunks", num_chunks)
    col3.metric("Time", f"{rrf.merge_time_ms:.0f}ms")

    if hasattr(rrf, 'query_contributions') and rrf.query_contributions:
        contrib_data = [
            {"Chunk": cid[:30] + "..." if len(cid) > 30 else cid, "Found By": ", ".join(qt)}
            for cid, qt in list(rrf.query_contributions.items())[:5]
        ]
        if contrib_data:
            st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)


def _render_graph_stage(graph_meta: dict) -> None:
    """Render GraphRAG stage details."""
    extracted = graph_meta.get("extracted_entities", [])
    matched = graph_meta.get("query_entities", [])
    communities = graph_meta.get("community_context", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Extracted", len(extracted))
    col2.metric("Matched", len(matched))
    col3.metric("Graph Chunks", graph_meta.get("graph_chunk_count", 0))
    col4.metric("Communities", len(communities))

    # Show extracted vs matched entities
    if extracted:
        st.markdown(f"**LLM Extracted:** {', '.join(extracted)}")
        if matched:
            st.markdown(f"**Found in Graph:** {', '.join(matched)}")
        else:
            st.caption("None of the extracted entities exist in the knowledge graph.")
    else:
        st.caption("LLM did not extract any entities from the query.")

    # Show community summaries
    if communities:
        st.markdown("**Relevant Communities:**")
        for comm in communities[:2]:
            st.info(f"{comm['summary'][:200]}...")


def _render_rerank_stage(rerank) -> None:
    """Render reranking stage details."""
    col1, col2 = st.columns(2)
    col1.markdown(f"**Model:** `{rerank.model}`")
    col2.metric("Time", f"{rerank.rerank_time_ms:.0f}ms")

    if rerank.order_changes:
        df = pd.DataFrame(rerank.order_changes)
        df = df[["before_rank", "after_rank", "before_score", "after_score", "text_preview"]]
        df.columns = ["Before", "After", "Old Score", "New Score", "Preview"]
        df["Old Score"] = df["Old Score"].round(3)
        df["New Score"] = df["New Score"].round(3)
        st.dataframe(df, use_container_width=True)


def _render_generation_stage(ans) -> None:
    """Render generation stage details."""
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Model:** `{ans.model}`")
    col2.metric("Time", f"{ans.generation_time_ms:.0f}ms")
    col3.markdown(f"**Sources:** {ans.sources_used}")

    with st.expander("Show Prompt", expanded=False):
        st.code(f"[System]\n{ans.system_prompt_used}\n\n[User]\n{ans.user_prompt_used}", language="text")


def _render_pipeline_log():
    """Render executed pipeline stages (only shows stages that were used)."""
    prep = st.session_state.preprocessed_query
    settings = st.session_state.retrieval_settings
    results = st.session_state.search_results
    rrf = st.session_state.rrf_data
    graph_meta = st.session_state.graph_metadata
    rerank = st.session_state.rerank_data
    ans = st.session_state.generated_answer

    # Preprocessing (only if used)
    if prep and getattr(prep, 'strategy_used', 'none') != 'none':
        with st.expander("Preprocessing", expanded=True):
            _render_preprocessing_stage(prep)

    # Retrieval (always shown when we have results)
    if settings:
        with st.expander("Retrieval", expanded=True):
            _render_retrieval_stage(settings, results, prep)

    # RRF (only if multi-query was used)
    if rrf:
        with st.expander("RRF Merging", expanded=False):
            _render_rrf_stage(rrf, prep)

    # GraphRAG (only if used and successful)
    if graph_meta and not graph_meta.get("error"):
        with st.expander("Graph Enrichment", expanded=False):
            _render_graph_stage(graph_meta)
    elif graph_meta and graph_meta.get("error"):
        st.warning(f"GraphRAG failed: {graph_meta['error']}")

    # Reranking (only if enabled)
    if rerank:
        with st.expander("Reranking", expanded=False):
            _render_rerank_stage(rerank)

    # Generation (always shown when we have an answer)
    if ans:
        with st.expander("Generation", expanded=True):
            _render_generation_stage(ans)


# ============================================================================
# SIDEBAR - User flow: Preprocessing → Collection → Retrieval → Reranking
# ============================================================================

st.sidebar.title("Settings")

# Load collections once (cached)
try:
    collection_infos = get_available_collections()
    st.session_state.connection_error = None
except Exception as e:
    collection_infos = []
    st.session_state.connection_error = str(e)

# -----------------------------------------------------------------------------
# Query Preprocessing (always visible - user selects strategy first)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Query Preprocessing")

# Build strategy options
strategy_options = {s[0]: (s[1], s[2]) for s in AVAILABLE_PREPROCESSING_STRATEGIES}
strategy_ids = list(strategy_options.keys())

# Get current index from session state
current_strategy = st.session_state.ui_preprocessing_strategy
current_idx = strategy_ids.index(current_strategy) if current_strategy in strategy_ids else 0

selected_strategy = st.sidebar.selectbox(
    "Strategy",
    options=strategy_ids,
    index=current_idx,
    format_func=lambda x: f"{strategy_options[x][0]} - {strategy_options[x][1]}",
    help="How to transform the query before searching.",
)

# Detect strategy change and reset collection
if selected_strategy != st.session_state.ui_preprocessing_strategy:
    st.session_state.ui_preprocessing_strategy = selected_strategy
    st.session_state.ui_collection = None  # Reset collection on strategy change
    st.rerun()

enable_preprocessing = selected_strategy != "none"

st.sidebar.divider()

# -----------------------------------------------------------------------------
# Collection (shown after preprocessing selected, hidden for graphrag)
# -----------------------------------------------------------------------------
if selected_strategy == "graphrag":
    # GraphRAG auto-selects section collection
    st.sidebar.markdown("### Collection")
    st.sidebar.caption("graphrag uses section chunks (required for entity matching)")
    selected_collection = _find_section_collection(collection_infos) if collection_infos else None
elif collection_infos:
    st.sidebar.markdown("### Collection")

    # Filter collections based on preprocessing compatibility
    compatible_collections = [
        info for info in collection_infos
        if selected_strategy in get_valid_preprocessing_strategies(info.strategy)
    ]

    if compatible_collections:
        collection_names = [info.collection_name for info in compatible_collections]
        collection_display = {info.collection_name: info.display_name for info in compatible_collections}

        # Get current index from session state
        current_coll = st.session_state.ui_collection
        current_coll_idx = collection_names.index(current_coll) if current_coll in collection_names else 0

        selected_collection = st.sidebar.selectbox(
            "Chunking Strategy",
            options=collection_names,
            index=current_coll_idx,
            format_func=lambda x: collection_display[x],
            help="Which chunking method to search.",
        )

        # Update session state if changed
        if selected_collection != st.session_state.ui_collection:
            st.session_state.ui_collection = selected_collection
    else:
        st.sidebar.warning("No compatible collections for this strategy.")
        selected_collection = None
else:
    st.sidebar.warning("No collections found. Is Weaviate running?")
    selected_collection = None

st.sidebar.divider()

# -----------------------------------------------------------------------------
# Retrieval (always visible)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Retrieval")

search_type = st.sidebar.radio(
    "Search Type",
    options=["vector", "hybrid"],
    index=1,
    format_func=lambda x: "Semantic" if x == "vector" else "Hybrid",
    help="Hybrid = vector + keyword matching.",
    horizontal=True,
)

if search_type == "hybrid":
    alpha = st.sidebar.slider(
        "Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="0 = keyword only, 1 = vector only",
    )
else:
    alpha = 0.5

top_k = st.sidebar.slider(
    "Results",
    min_value=1,
    max_value=MAX_TOP_K,
    value=DEFAULT_TOP_K,
    help="Number of chunks to retrieve.",
)

st.sidebar.divider()

# -----------------------------------------------------------------------------
# Reranking (always visible)
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Reranking")

use_reranking = st.sidebar.checkbox(
    "Enable Cross-Encoder",
    value=False,
    help="Re-score results for higher accuracy (slow on CPU).",
)

if use_reranking:
    st.sidebar.caption("~2 min/query on CPU")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("RAGLab Search")
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
                    preprocessed = preprocess_query(
                        query, model=PREPROCESSING_MODEL, strategy=selected_strategy
                    )
                    search_query = preprocessed.search_query
                    st.session_state.preprocessed_query = preprocessed
                except Exception as e:
                    st.warning(f"Preprocessing failed: {e}. Using original query.")
                    st.session_state.preprocessed_query = None
        else:
            st.session_state.preprocessed_query = None

        # Step 2 & 3: Search (with optional reranking and RRF for multi-query)
        # Check if multi-query strategy was used
        multi_queries = None
        if preprocessed and preprocessed.generated_queries:
            multi_queries = preprocessed.generated_queries

        spinner_msg = "Stage 2: Searching..."
        if multi_queries and len(multi_queries) > 1:
            spinner_msg = f"Stage 2: Searching ({len(multi_queries)} queries + RRF)..."
        if use_reranking:
            spinner_msg = spinner_msg.replace("...", " + reranking...")

        with st.spinner(spinner_msg):
            try:
                search_output = search_chunks(
                    query=search_query,
                    top_k=top_k,
                    search_type=search_type,
                    alpha=alpha,
                    collection_name=selected_collection,
                    use_reranking=use_reranking,
                    multi_queries=multi_queries,
                    strategy=selected_strategy,
                )
                st.session_state.search_results = search_output.results
                st.session_state.rerank_data = search_output.rerank_data
                st.session_state.rrf_data = search_output.rrf_data
                st.session_state.graph_metadata = search_output.graph_metadata
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
                st.session_state.rrf_data = None
                st.session_state.graph_metadata = None
                preprocessed = None

        # Step 4: Answer Generation
        if st.session_state.search_results:
            with st.spinner("Stage 4: Generating answer..."):
                try:
                    answer = generate_answer(
                        query=query,
                        chunks=st.session_state.search_results,
                        model=GENERATION_MODEL,
                    )
                    st.session_state.generated_answer = answer
                except Exception as e:
                    st.warning(f"Answer generation failed: {e}")
                    st.session_state.generated_answer = None
        else:
            # No results to generate from
            st.session_state.generated_answer = None

        # Auto-save successful queries to log
        if st.session_state.search_results:
            from src.shared.query_logger import log_query
            log_query(
                query=query,
                preprocessed=preprocessed,
                retrieval_settings=st.session_state.retrieval_settings,
                search_results=st.session_state.search_results,
                rerank_data=st.session_state.rerank_data,
                generated_answer=st.session_state.generated_answer,
                collection_name=selected_collection,
                rrf_data=st.session_state.rrf_data,
            )


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
                st.markdown("#### Query Preprocessing")
                col1, col2 = st.columns(2)

                strategy_used = getattr(prep, 'strategy_used', 'N/A')
                col1.markdown(f"**Strategy:** `{strategy_used}`")
                col2.markdown(f"**Time:** {prep.preprocessing_time_ms:.0f}ms")

                # Show HyDE passage if applied
                hyde_passage = getattr(prep, 'hyde_passage', None)
                if hyde_passage and hyde_passage != prep.original_query:
                    st.info(f"**HyDE Passage:** {hyde_passage[:100]}...")

                # Show multi-query info
                generated_queries = getattr(prep, 'generated_queries', None)
                if generated_queries and len(generated_queries) > 1:
                    st.info(f"**Multi-Query:** {len(generated_queries)} queries generated")

                # Show decomposition info
                sub_queries = getattr(prep, 'sub_queries', None)
                if sub_queries and len(sub_queries) > 0:
                    st.info(f"**Decomposed into:** {len(sub_queries)} sub-questions")

                st.divider()

        # Generated Answer Section
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

            # Display formatted references for cited sources
            results = st.session_state.search_results
            if ans.sources_used and results:
                st.markdown("---")
                st.caption("References")
                for idx in sorted(ans.sources_used):
                    if 1 <= idx <= len(results):
                        chunk = results[idx - 1]  # Convert 1-based to 0-based
                        book_parts = chunk["book_id"].rsplit("(", 1)
                        book_title = book_parts[0].strip()
                        author = book_parts[1].rstrip(")") if len(book_parts) > 1 else ""
                        section = chunk.get("section", "")
                        ref_text = f"[{idx}] {book_title}"
                        if author:
                            ref_text += f" — {author}"
                        if section:
                            ref_text += f", Section: {section}"
                        st.caption(ref_text)

        else:
            st.warning("Answer generation failed. Check the Pipeline Log tab for details.")

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

        # Show score explanation based on retrieval method
        prep = st.session_state.preprocessed_query
        strategy = getattr(prep, 'strategy_used', 'none') if prep else 'none'
        reranked = st.session_state.get("rerank_data") is not None

        if reranked:
            score_info = "Scores: cross-encoder semantic relevance (0.0–1.0+, higher = more relevant)"
        elif st.session_state.get("rrf_data") is not None:
            score_info = "Scores: RRF (Reciprocal Rank Fusion, k=60). Range ~0.01–0.05. See: Cormack et al. (2009)"
        else:
            score_info = "Scores: cosine similarity (0.0–1.0, higher = more semantically similar)"

        st.caption(score_info)
        _display_chunks(st.session_state.search_results)

elif query and not st.session_state.search_results:
    st.info("No results found. Try a different query.")

else:
    st.info("Enter a query above to search the knowledge base.")


