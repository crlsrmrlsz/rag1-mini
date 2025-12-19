"""RAG1-Mini Search Interface.

A Streamlit application for testing the RAG system with Weaviate backend.

Run with:
    streamlit run src/ui/app.py

Prerequisites:
    - Weaviate must be running (docker compose up -d)
    - Stage 6 must have been run to populate the collection
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.config import BOOK_CATEGORIES, DEFAULT_TOP_K, MAX_TOP_K
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

# Search type selector
search_type = st.sidebar.radio(
    "Search Type",
    options=["vector", "hybrid"],
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

# Book selection
st.sidebar.subheader("Book Filters")

# Quick select buttons
col1, col2, col3 = st.sidebar.columns(3)

all_books = BOOK_CATEGORIES["neuroscience"] + BOOK_CATEGORIES["philosophy"]

if col1.button("All", use_container_width=True):
    st.session_state.selected_neuro = BOOK_CATEGORIES["neuroscience"]
    st.session_state.selected_phil = BOOK_CATEGORIES["philosophy"]
    st.rerun()

if col2.button("Neuro", use_container_width=True):
    st.session_state.selected_neuro = BOOK_CATEGORIES["neuroscience"]
    st.session_state.selected_phil = []
    st.rerun()

if col3.button("Phil", use_container_width=True):
    st.session_state.selected_neuro = []
    st.session_state.selected_phil = BOOK_CATEGORIES["philosophy"]
    st.rerun()

# Initialize selected books if not set
if "selected_neuro" not in st.session_state:
    st.session_state.selected_neuro = BOOK_CATEGORIES["neuroscience"]
if "selected_phil" not in st.session_state:
    st.session_state.selected_phil = BOOK_CATEGORIES["philosophy"]

# Neuroscience books multiselect
st.sidebar.markdown("**Neuroscience**")
neuro_selection = st.sidebar.multiselect(
    "Neuroscience Books",
    options=BOOK_CATEGORIES["neuroscience"],
    default=st.session_state.selected_neuro,
    label_visibility="collapsed",
    key="neuro_select",
)

# Philosophy books multiselect
st.sidebar.markdown("**Philosophy**")
phil_selection = st.sidebar.multiselect(
    "Philosophy Books",
    options=BOOK_CATEGORIES["philosophy"],
    default=st.session_state.selected_phil,
    label_visibility="collapsed",
    key="phil_select",
)

selected_books = neuro_selection + phil_selection


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
col1, col2 = st.columns([1, 4])
with col1:
    search_clicked = st.button("Search", type="primary", disabled=not query)
with col2:
    if selected_books:
        st.caption(f"Searching {len(selected_books)} of {len(all_books)} books")
    else:
        st.caption("No books selected")

# Execute search
if search_clicked and query:
    if not selected_collection:
        st.error("No collection available. Please run `docker compose up -d` and run Stage 6.")
    elif not selected_books:
        st.warning("Please select at least one book to search.")
    else:
        with st.spinner("Searching..."):
            try:
                # Use filter only if not all books selected
                book_filter = selected_books if len(selected_books) < len(all_books) else None

                results = search_chunks(
                    query=query,
                    book_filter=book_filter,
                    top_k=top_k,
                    search_type=search_type,
                    alpha=alpha,
                    collection_name=selected_collection,
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
    st.info("No results found. Try a different query or select more books.")

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

    3. **Filtering** optionally restricts results to selected books.

    4. **Results** show the most relevant text passages with their source.

    ### Search Types

    - **Semantic (Vector)**: Finds chunks with similar *meaning*, even if
      they use different words. Good for concepts and ideas.

    - **Hybrid**: Combines semantic search with keyword matching (BM25).
      Good for technical terms that should match exactly.

    ### Technical Details

    - **Embedding Model**: text-embedding-3-large (3072 dimensions)
    - **Vector Database**: Weaviate with HNSW index
    - **Distance Metric**: Cosine similarity (1.0 = identical)
    - **Chunk Size**: ~800 tokens with 2-sentence overlap

    ### Python Concepts in This UI

    - **Session State**: `st.session_state` persists data across UI interactions
    - **Reactive Updates**: Streamlit reruns the script when inputs change
    - **Component Layout**: Sidebars, columns, expanders for organization
    - **Error Handling**: Try/except with user-friendly error messages
    """)
