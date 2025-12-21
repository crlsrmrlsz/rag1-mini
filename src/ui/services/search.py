"""Weaviate search service for RAG UI.

Provides semantic search with optional cross-encoder reranking.
This module wraps the vector_db query functions for use in the Streamlit UI.

## Search Types

1. **Vector (Semantic)**: Uses embedding similarity to find related content.
   Best for conceptual queries like "What is consciousness?"

2. **Hybrid (Vector + Keyword)**: Combines embedding similarity with BM25 keyword
   matching. Best for queries with specific terms like "thalamocortical hub".

## Reranking

Optional cross-encoder reranking improves result quality by processing query
and document together. The cross-encoder sees both texts simultaneously,
enabling deeper semantic understanding than bi-encoders (embeddings).

Two-Stage Retrieval:
1. Fast bi-encoder retrieves top-50 candidates
2. Slow cross-encoder reranks to top-k with higher accuracy
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from src.config import get_collection_name, DEFAULT_TOP_K
from src.vector_db import get_client, query_similar, query_hybrid, SearchResult

# Default number of candidates to retrieve before reranking
# Higher = more accurate but slower (50 is a good balance)
RERANK_INITIAL_K = 50


@dataclass
class SearchOutput:
    """Result of search operation including optional rerank data for logging.

    Attributes:
        results: List of chunk dictionaries.
        rerank_data: If reranking was used, contains RerankResult for logging.
    """
    results: List[Dict[str, Any]] = field(default_factory=list)
    rerank_data: Optional[Any] = None  # RerankResult when reranking is used


def search_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    search_type: str = "vector",
    alpha: float = 0.5,
    collection_name: Optional[str] = None,
    use_reranking: bool = False,
) -> SearchOutput:
    """
    Search Weaviate for relevant chunks with optional reranking.

    This is the main search function for the UI. It connects to Weaviate,
    executes the search, optionally applies cross-encoder reranking, and
    returns results as dictionaries.

    Args:
        query: User's search query.
        top_k: Number of results to return.
        search_type: Either "vector" (semantic) or "hybrid" (vector + keyword).
        alpha: For hybrid search, balance between vector (1.0) and keyword (0.0).
        collection_name: Override collection (for future multi-collection).
        use_reranking: If True, apply cross-encoder reranking for better accuracy.
                       This is slower but significantly improves result quality.

    Returns:
        SearchOutput with results list and optional rerank_data for logging.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If Weaviate is not running.

    Example:
        >>> # Basic hybrid search
        >>> output = search_chunks("What is consciousness?", search_type="hybrid")
        >>> results = output.results
        >>>
        >>> # With reranking for higher accuracy
        >>> output = search_chunks("What is consciousness?", use_reranking=True)
        >>> print(output.rerank_data.order_changes)  # See how rankings changed
    """
    collection_name = collection_name or get_collection_name()
    rerank_data = None

    # Connect to Weaviate
    client = get_client()

    try:
        # Determine how many candidates to retrieve
        # If reranking, get more candidates for the reranker to work with
        initial_k = RERANK_INITIAL_K if use_reranking else top_k

        # Execute search based on type
        if search_type == "hybrid":
            results: List[SearchResult] = query_hybrid(
                client=client,
                query_text=query,
                top_k=initial_k,
                alpha=alpha,
                collection_name=collection_name,
            )
        else:
            results: List[SearchResult] = query_similar(
                client=client,
                query_text=query,
                top_k=initial_k,
                collection_name=collection_name,
            )

        # Apply cross-encoder reranking if enabled
        if use_reranking and results:
            # Import here to avoid loading 1.2GB model on startup
            from src.reranking import rerank
            rerank_result = rerank(query, results, top_k=top_k)
            results = rerank_result.results
            rerank_data = rerank_result

        # Convert SearchResult objects to dicts for Streamlit
        result_dicts = [
            {
                "chunk_id": r.chunk_id,
                "book_id": r.book_id,
                "section": r.section,
                "context": r.context,
                "text": r.text,
                "token_count": r.token_count,
                "similarity": r.score,
            }
            for r in results
        ]

        return SearchOutput(results=result_dicts, rerank_data=rerank_data)

    finally:
        client.close()


def list_collections() -> List[str]:
    """
    List all available RAG collections in Weaviate.

    Returns:
        List of collection names starting with 'RAG_'.
    """
    client = get_client()

    try:
        all_collections = client.collections.list_all()
        return sorted([name for name in all_collections.keys() if name.startswith("RAG_")])
    finally:
        client.close()
