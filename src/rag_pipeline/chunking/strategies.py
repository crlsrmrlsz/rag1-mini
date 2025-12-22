"""Chunking strategy implementations.

## RAG Theory: Strategy Pattern for Chunking

Different chunking strategies optimize for different retrieval scenarios:
- section: Sequential reading order with overlap (baseline, fast)
- semantic: Embedding similarity-based boundaries (improved coherence)
- contextual: LLM-generated chunk context (Anthropic-style, +35% improvement)
- raptor: Hierarchical summarization tree (multi-level retrieval)

The strategy pattern allows A/B testing chunking approaches without modifying
the pipeline. Each strategy outputs to its own subdirectory for isolated
Weaviate collections.

## Library Usage

Wraps existing chunking functions (section_chunker, semantic_chunker, etc.)
in a common interface for the stage runner to invoke.

## Data Flow

1. User selects strategy (CLI arg: --strategy semantic)
2. Stage 4 runner calls get_strategy() to get the function
3. Strategy function processes all files from DIR_NLP_CHUNKS
4. Outputs to DIR_FINAL_CHUNKS/{strategy_name}/
5. Returns stats dict {book_name: chunk_count}
"""

from functools import partial
from typing import Any, Callable, Dict, List

from src.config import MAX_CHUNK_TOKENS, OVERLAP_SENTENCES, SEMANTIC_SIMILARITY_THRESHOLD
from src.shared.files import setup_logging

logger = setup_logging(__name__)


# Type alias for strategy functions
# Input: None (reads from DIR_NLP_CHUNKS)
# Output: Dict[book_name, chunk_count]
ChunkingStrategyFunction = Callable[[], Dict[str, int]]


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================


def section_strategy() -> Dict[str, int]:
    """Sequential chunking with sentence overlap (baseline).

    Algorithm:
    - Read paragraphs in document order
    - Build chunks up to MAX_CHUNK_TOKENS
    - Overlap OVERLAP_SENTENCES between chunks (same section only)
    - Respects markdown section boundaries (# Chapter, ## Section)

    Use case: Preserves reading order, best for linear narratives.
    Fast execution, no API calls during chunking.

    Returns:
        Dict mapping book names to chunk counts.
    """
    from src.rag_pipeline.chunking.section_chunker import run_section_chunking

    logger.info(f"[section] Using sequential chunking with overlap")
    logger.info(f"[section] Max tokens: {MAX_CHUNK_TOKENS}, overlap: {OVERLAP_SENTENCES}")
    return run_section_chunking()


def semantic_strategy(
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> Dict[str, int]:
    """Semantic similarity-based chunking.

    Algorithm:
    - Embed sentences using text-embedding-3-large API
    - Compute cosine similarity between adjacent sentences
    - Split where similarity drops below threshold (topic shift)
    - Still respects section boundaries and token limits
    - Uses same overlap mechanism as section strategy

    Use case: Better topical coherence, improved retrieval precision.
    Research shows 8-12% improvement on Q&A tasks.

    Note: Requires API calls during chunking (costs apply).

    Args:
        similarity_threshold: Cosine similarity threshold (0.0-1.0) for detecting
            topic shifts. Lower = fewer splits (larger chunks). Default from config.

    Returns:
        Dict mapping book names to chunk counts.
    """
    from src.rag_pipeline.chunking.semantic_chunker import run_semantic_chunking

    logger.info(f"[semantic] Using embedding similarity chunking")
    logger.info(f"[semantic] Max tokens: {MAX_CHUNK_TOKENS}, threshold: {similarity_threshold}")
    return run_semantic_chunking(similarity_threshold=similarity_threshold)


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================


STRATEGIES: Dict[str, ChunkingStrategyFunction] = {
    "section": section_strategy,
    "semantic": semantic_strategy,
}


def get_strategy(strategy_id: str, **kwargs: Any) -> ChunkingStrategyFunction:
    """Get chunking strategy function by ID.

    Args:
        strategy_id: One of "section", "semantic" (future: "contextual", "raptor").
        **kwargs: Optional parameters to pass to the strategy function.
            For semantic strategy: similarity_threshold (float).

    Returns:
        Strategy function that takes no args and returns Dict[str, int].

    Raises:
        ValueError: If strategy_id is not registered.

    Example:
        >>> strategy_fn = get_strategy("semantic", similarity_threshold=0.6)
        >>> stats = strategy_fn()
        >>> print(stats)  # {"book1": 45, "book2": 67}
    """
    if strategy_id not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown chunking strategy '{strategy_id}'. Available: {available}")

    strategy_fn = STRATEGIES[strategy_id]
    if kwargs:
        return partial(strategy_fn, **kwargs)
    return strategy_fn


def list_strategies() -> List[str]:
    """List all registered strategy IDs.

    Returns:
        List of strategy IDs (e.g., ["section", "semantic"]).
    """
    return list(STRATEGIES.keys())
