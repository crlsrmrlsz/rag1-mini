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
)
from src.rag_pipeline.indexing import get_client, query_similar, query_hybrid, SearchResult

# Default number of candidates to retrieve before reranking
# Higher = more accurate but slower (50 is a good balance)
RERANK_INITIAL_K = 50

# Import RRF for multi-query merging
from src.rag_pipeline.retrieval.rrf import reciprocal_rank_fusion, RRFResult


@dataclass
class SearchOutput:
    """Result of search operation including optional rerank and RRF data.

    Attributes:
        results: List of chunk dictionaries.
        rerank_data: If reranking was used, contains RerankResult for logging.
        rrf_data: If RRF merging was used, contains RRFResult for logging.
    """
    results: List[Dict[str, Any]] = field(default_factory=list)
    rerank_data: Optional[Any] = None  # RerankResult when reranking is used
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
            from src.rag_pipeline.retrieval.reranking import rerank
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
                from src.rag_pipeline.retrieval.reranking import rerank
                rerank_result = rerank(query, results, top_k=top_k)
                results = rerank_result.results
                rerank_data = rerank_result

        finally:
            client.close()

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


# ============================================================================
# COLLECTION METADATA ENRICHMENT
# ============================================================================

from src.config import get_strategy_metadata, StrategyMetadata


@dataclass
class CollectionInfo:
    """Enriched collection metadata for UI display.

    Attributes:
        collection_name: Full Weaviate collection name (e.g., "RAG_section_embed3large_v1").
        strategy: Strategy key extracted from collection name (e.g., "section").
        display_name: Human-readable name for UI (e.g., "Section-Based Chunking").
        description: Short description of the strategy.
        is_available: Whether the collection exists in Weaviate.
    """
    collection_name: str
    strategy: str
    display_name: str
    description: str
    is_available: bool


def extract_strategy_from_collection(collection_name: str) -> str:
    """
    Extract strategy key from collection name.

    Args:
        collection_name: Collection name like "RAG_section_embed3large_v1"
                        or "RAG_semantic_0.5_embed3large_v1".

    Returns:
        Strategy key like "section" or "semantic_0.5".

    Example:
        >>> extract_strategy_from_collection("RAG_section_embed3large_v1")
        'section'
        >>> extract_strategy_from_collection("RAG_semantic_0.5_embed3large_v1")
        'semantic_0.5'
        >>> extract_strategy_from_collection("RAG_contextual_embed3large_v1")
        'contextual'
    """
    # Format: RAG_{strategy}_{model}_v{version}
    # Strategy can be: "section", "contextual", "semantic_0.5", "semantic_0.75"
    if not collection_name.startswith("RAG_"):
        return "unknown"

    # Remove RAG_ prefix
    rest = collection_name[4:]

    # Find the model suffix pattern (embed3large or similar)
    # Strategy ends before the model suffix
    parts = rest.split("_")

    if len(parts) < 3:
        return "unknown"

    # Check for semantic_X.X pattern (strategy is 2 parts)
    if parts[0] == "semantic" and len(parts) >= 2:
        # Try to parse the second part as a threshold
        try:
            float(parts[1])
            return f"{parts[0]}_{parts[1]}"
        except ValueError:
            pass  # Not a threshold, treat as single-part strategy

    # Single-part strategy
    return parts[0]


def get_available_collections() -> List[CollectionInfo]:
    """
    List all RAG collections with enriched metadata.

    Queries Weaviate for available collections, then enriches each with
    metadata from the strategy registry for UI display.

    Returns:
        List of CollectionInfo objects with display names and descriptions.

    Example:
        >>> collections = get_available_collections()
        >>> for c in collections:
        ...     print(f"{c.display_name}: {c.description}")
        Section-Based Chunking: Preserves document structure with sentence overlap
        Contextual Chunking: LLM-generated context prepended (+35% improvement)
    """
    existing_collections = list_collections()

    collection_infos = []
    for coll_name in existing_collections:
        strategy = extract_strategy_from_collection(coll_name)
        metadata = get_strategy_metadata(strategy)

        collection_infos.append(CollectionInfo(
            collection_name=coll_name,
            strategy=strategy,
            display_name=metadata.display_name,
            description=metadata.description,
            is_available=True,
        ))

    return collection_infos
