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

from src.config import (
    get_collection_name,
    DEFAULT_TOP_K,
    ENABLE_DIVERSITY_BALANCING,
    DIVERSITY_MIN_SCORE,
    DIVERSITY_BALANCE,
)
from src.vector_db import get_client, query_similar, query_hybrid, SearchResult

# Default number of candidates to retrieve before reranking
# Higher = more accurate but slower (50 is a good balance)
RERANK_INITIAL_K = 50

# Import RRF for multi-query merging
from src.retrieval import reciprocal_rank_fusion, RRFResult


@dataclass
class SearchOutput:
    """Result of search operation including optional rerank, RRF, and diversity data.

    Attributes:
        results: List of chunk dictionaries.
        rerank_data: If reranking was used, contains RerankResult for logging.
        diversity_data: If diversity balancing was applied, contains DiversityResult.
        rrf_data: If RRF merging was used, contains RRFResult for logging.
    """
    results: List[Dict[str, Any]] = field(default_factory=list)
    rerank_data: Optional[Any] = None  # RerankResult when reranking is used
    diversity_data: Optional[Any] = None  # DiversityResult when balancing is used
    rrf_data: Optional[Any] = None  # RRFResult when multi-query is used


def search_multi_query(
    queries: List[Dict[str, str]],
    top_k: int = DEFAULT_TOP_K,
    search_type: str = "hybrid",
    alpha: float = 0.5,
    collection_name: Optional[str] = None,
) -> Tuple[List[SearchResult], RRFResult]:
    """Execute multiple queries and merge with Reciprocal Rank Fusion.

    Each query is executed independently, then results are merged using RRF
    to produce a single ranked list that benefits from query diversity.

    Args:
        queries: List of {type, query} dicts from multi_query_strategy.
        top_k: Final number of results after merging.
        search_type: "vector" or "hybrid".
        alpha: Hybrid search alpha.
        collection_name: Weaviate collection.

    Returns:
        Tuple of (merged SearchResult list, RRFResult for logging).
    """
    collection_name = collection_name or get_collection_name()
    client = get_client()

    try:
        result_lists = []
        query_types = []

        # Retrieve more per query to give RRF enough candidates
        per_query_k = max(top_k * 2, 20)

        for q in queries:
            query_text = q.get("query", "")
            query_type = q.get("type", "unknown")

            if not query_text:
                continue

            if search_type == "hybrid":
                results = query_hybrid(
                    client=client,
                    query_text=query_text,
                    top_k=per_query_k,
                    alpha=alpha,
                    collection_name=collection_name,
                )
            else:
                results = query_similar(
                    client=client,
                    query_text=query_text,
                    top_k=per_query_k,
                    collection_name=collection_name,
                )

            result_lists.append(results)
            query_types.append(query_type)

        # Merge with RRF
        rrf_result = reciprocal_rank_fusion(
            result_lists=result_lists,
            query_types=query_types,
            top_k=top_k,
        )

        return rrf_result.results, rrf_result

    finally:
        client.close()


def search_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    search_type: str = "vector",
    alpha: float = 0.5,
    collection_name: Optional[str] = None,
    use_reranking: bool = False,
    multi_queries: Optional[List[Dict[str, str]]] = None,
) -> SearchOutput:
    """
    Search Weaviate for relevant chunks with optional reranking and RRF merging.

    This is the main search function for the UI. It connects to Weaviate,
    executes the search (or multiple searches for multi-query), optionally
    applies cross-encoder reranking, and returns results as dictionaries.

    Args:
        query: User's search query (used for reranking even in multi-query mode).
        top_k: Number of results to return.
        search_type: Either "vector" (semantic) or "hybrid" (vector + keyword).
        alpha: For hybrid search, balance between vector (1.0) and keyword (0.0).
        collection_name: Override collection (for future multi-collection).
        use_reranking: If True, apply cross-encoder reranking for better accuracy.
                       This is slower but significantly improves result quality.
        multi_queries: If provided, execute all queries and merge with RRF.
                       Expected format: [{"type": "neuro", "query": "..."}, ...]

    Returns:
        SearchOutput with results list and optional rerank_data/rrf_data for logging.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If Weaviate is not running.

    Example:
        >>> # Basic hybrid search
        >>> output = search_chunks("What is consciousness?", search_type="hybrid")
        >>> results = output.results
        >>>
        >>> # With multi-query RRF merging
        >>> queries = [{"type": "neuro", "query": "..."}, {"type": "philo", "query": "..."}]
        >>> output = search_chunks("What is consciousness?", multi_queries=queries)
        >>> print(output.rrf_data.merge_time_ms)  # See RRF performance
    """
    collection_name = collection_name or get_collection_name()
    rerank_data = None
    rrf_data = None

    # Multi-query path: execute all queries and merge with RRF
    if multi_queries and len(multi_queries) > 1:
        # Get more candidates for reranking to work with
        initial_k = RERANK_INITIAL_K if use_reranking else top_k

        results, rrf_data = search_multi_query(
            queries=multi_queries,
            top_k=initial_k,
            search_type=search_type,
            alpha=alpha,
            collection_name=collection_name,
        )

        # Apply cross-encoder reranking to merged results
        if use_reranking and results:
            from src.reranking import rerank
            rerank_result = rerank(query, results, top_k=top_k)
            results = rerank_result.results
            rerank_data = rerank_result

    else:
        # Single-query path (original behavior)
        client = get_client()

        try:
            # Determine how many candidates to retrieve
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
                from src.reranking import rerank
                rerank_result = rerank(query, results, top_k=top_k)
                results = rerank_result.results
                rerank_data = rerank_result

        finally:
            client.close()

    # Apply diversity balancing if enabled (works for both paths)
    diversity_data = None
    if ENABLE_DIVERSITY_BALANCING and results:
        from src.diversity import apply_diversity_balance
        diversity_result = apply_diversity_balance(
            results=results,
            target_count=top_k,
            balance=DIVERSITY_BALANCE,
            min_score=DIVERSITY_MIN_SCORE,
        )
        results = diversity_result.results
        diversity_data = diversity_result

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

    return SearchOutput(
        results=result_dicts,
        rerank_data=rerank_data,
        diversity_data=diversity_data,
        rrf_data=rrf_data,
    )


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
