"""Shared utilities for reranking operations.

## RAG Theory: Two-Stage Retrieval

Cross-encoder reranking is the second stage of a two-stage retrieval system:
1. **Stage 1 (Bi-encoder)**: Fast retrieval with embeddings (50 candidates)
2. **Stage 2 (Cross-encoder)**: Slower but more accurate reranking to top-k

Cross-encoders see query and document together, enabling deeper semantic
understanding than bi-encoders which embed them separately.

## Library Usage

Uses the reranking module from src.rag_pipeline.retrieval.reranking which
wraps the cross-encoder model for scoring query-document pairs.

## Data Flow

SearchResult list → apply_reranking_if_enabled() → Filtered list (List[SearchResult])

For UI code that needs metadata (timing, order changes):
SearchResult list → apply_reranking_with_metadata() → (List[SearchResult], RerankResult | None)

This module consolidates reranking logic used across evaluation and UI
to ensure consistent behavior and reduce code duplication.
"""

from typing import Optional

from src.rag_pipeline.indexing.weaviate_query import SearchResult
from src.rag_pipeline.retrieval.reranking import rerank, RerankResult


def apply_reranking_if_enabled(
    results: list[SearchResult],
    question: str,
    top_k: int,
    use_reranking: bool,
) -> list[SearchResult]:
    """Apply cross-encoder reranking if enabled and results exist.

    This is the simple interface for evaluation code that only needs
    the reranked results list.

    Args:
        results: Search results to potentially rerank.
        question: Original question for reranking context.
        top_k: Number of results to return after reranking.
        use_reranking: Whether reranking is enabled.

    Returns:
        Reranked list if use_reranking is True and results exist,
        otherwise returns original results.

    Example:
        >>> results = query_hybrid(client, "What is consciousness?", top_k=50)
        >>> # With reranking enabled
        >>> reranked = apply_reranking_if_enabled(results, "What is consciousness?", 10, True)
        >>> len(reranked) <= 10
        True
        >>> # With reranking disabled
        >>> same_results = apply_reranking_if_enabled(results, "query", 10, False)
        >>> same_results == results
        True
    """
    if use_reranking and results:
        return rerank(question, results, top_k=top_k).results
    return results


def apply_reranking_with_metadata(
    results: list[SearchResult],
    question: str,
    top_k: int,
    use_reranking: bool,
) -> tuple[list[SearchResult], Optional[RerankResult]]:
    """Apply cross-encoder reranking and return both results and metadata.

    This is the interface for UI code that needs reranking metadata
    (timing info, order changes) for logging/display.

    Args:
        results: Search results to potentially rerank.
        question: Original question for reranking context.
        top_k: Number of results to return after reranking.
        use_reranking: Whether reranking is enabled.

    Returns:
        Tuple of (reranked_results, rerank_metadata):
        - reranked_results: List of SearchResult (reranked or original)
        - rerank_metadata: RerankResult with timing/order info, or None if not reranked

    Example:
        >>> results = query_hybrid(client, "What is consciousness?", top_k=50)
        >>> reranked, metadata = apply_reranking_with_metadata(results, "question", 10, True)
        >>> if metadata:
        ...     print(f"Reranking took {metadata.rerank_time_ms:.1f}ms")
    """
    if use_reranking and results:
        rerank_result = rerank(question, results, top_k=top_k)
        return rerank_result.results, rerank_result
    return results, None
