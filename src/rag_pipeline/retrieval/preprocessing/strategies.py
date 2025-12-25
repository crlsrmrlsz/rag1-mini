"""Preprocessing strategy implementations.

## RAG Theory: Strategy Pattern for Query Preprocessing

Each strategy applies its transformation directly to any query, following
the original research papers' design. The strategies don't use classification
routing - they simply transform queries for better retrieval.

Strategies:
- none: No transformation (baseline for comparison)
- hyde: Generate hypothetical answer for semantic matching (arXiv:2212.10496)
- decomposition: Break into sub-questions + RRF merge (+36.7% MRR@10, arXiv:2507.00355)
- graphrag: Hybrid graph + vector retrieval via RRF (arXiv:2404.16130)

The strategy pattern allows easy A/B testing and adding new strategies
without modifying existing code.

## Library Usage

Uses the existing query_preprocessing functions (hyde_prompt, etc.)
wrapped in strategy functions that conform to a common signature.

## Data Flow

1. User selects strategy (UI dropdown, CLI arg, or config default)
2. preprocess_query() calls get_strategy() to get the strategy function
3. Strategy function processes query and returns PreprocessedQuery
4. Result includes strategy_used field for tracking
"""

import time
from typing import Callable, Dict, List, Optional

from src.config import PREPROCESSING_MODEL
from src.rag_pipeline.retrieval.preprocessing.query_preprocessing import (
    PreprocessedQuery,
    hyde_prompt as _hyde_prompt_fn,
    decompose_query,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


# Type alias for strategy functions
StrategyFunction = Callable[[str, Optional[str]], PreprocessedQuery]


def none_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """No preprocessing - return original query unchanged.

    Use case: Baseline comparison for evaluation. Returns the query exactly
    as entered, with no LLM calls. Fastest strategy.

    Args:
        query: The user's original query.
        model: Ignored (no LLM call made).

    Returns:
        PreprocessedQuery with original query as search_query.
    """
    return PreprocessedQuery(
        original_query=query,
        search_query=query,
        strategy_used="none",
        preprocessing_time_ms=0.0,
        model=model or PREPROCESSING_MODEL,
    )


def hyde_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """HyDE: Generate hypothetical answer, use for retrieval.

    Hypothetical Document Embeddings (HyDE) generates a plausible answer
    to the query, then searches for real passages similar to this answer.
    This bridges the semantic gap between question embeddings and document
    embeddings.

    Research: arXiv:2212.10496 - Outperforms unsupervised dense retrievers

    Args:
        query: The user's original query.
        model: Model for HyDE LLM call.

    Returns:
        PreprocessedQuery with hypothetical passage as search_query.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Generate hypothetical answer
    hyde_passage = _hyde_prompt_fn(query, model=model)
    logger.info(f"[hyde] Generated hypothetical: {hyde_passage[:80]}...")

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        search_query=hyde_passage,  # Search with hypothetical answer!
        hyde_passage=hyde_passage,
        strategy_used="hyde",
        preprocessing_time_ms=elapsed_ms,
        model=model,
    )


# =============================================================================
# DECOMPOSITION STRATEGY
# =============================================================================


def decomposition_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Always decompose query into sub-questions for RRF merging.

    This strategy handles comparison and multi-aspect questions by:
    1. Decomposing into 2-4 sub-questions
    2. Using sub-questions for RRF-merged retrieval

    Based on Query Decomposition research showing +36.7% MRR@10 improvement
    for complex multi-hop queries. Works safely on simpler queries too.

    Args:
        query: The user's original query.
        model: Model for LLM calls.

    Returns:
        PreprocessedQuery with sub_queries and generated_queries for RRF.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Decompose query into sub-questions
    sub_queries, decomposition_response = decompose_query(query, model=model)
    logger.info(f"[decomposition] Decomposed into {len(sub_queries)} sub-queries")

    # Build generated_queries format for search compatibility
    # This allows reuse of existing RRF merging infrastructure
    generated_queries = [{"type": "original", "query": query}]
    for i, sq in enumerate(sub_queries):
        generated_queries.append({"type": f"sub_{i+1}", "query": sq})

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        search_query=query,  # Keep original for display
        sub_queries=sub_queries,
        strategy_used="decomposition",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        generated_queries=generated_queries,  # For search/RRF compatibility
        decomposition_response=decomposition_response,
    )


# =============================================================================
# GRAPHRAG STRATEGY
# =============================================================================


def graphrag_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """GraphRAG: Hybrid graph + vector retrieval.

    This strategy extracts entities from the query and sets up metadata
    for graph traversal during retrieval. The actual graph retrieval
    happens in the search phase (not preprocessing) where it can access
    the Neo4j driver.

    Key insight: GraphRAG is primarily a retrieval-time strategy, not a
    query transformation strategy. This function prepares the query with
    entity hints that the retrieval layer uses for graph traversal.

    Research: arXiv:2404.16130 - GraphRAG: +72-83% win rate vs baseline

    Args:
        query: The user's original query.
        model: Model for entity extraction (optional).

    Returns:
        PreprocessedQuery with graphrag metadata for retrieval layer.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Import here to avoid circular dependency
    from src.graph.query import extract_query_entities

    # Extract entities from query (no Neo4j lookup at this stage)
    query_entities = extract_query_entities(query, driver=None)
    logger.info(f"[graphrag] Extracted entities: {query_entities}")

    elapsed_ms = (time.time() - start_time) * 1000

    # Store entity hints in generated_queries for retrieval layer
    generated_queries = [
        {"type": "original", "query": query},
    ]
    # Add entity-focused queries for graph traversal hints
    for entity in query_entities[:5]:  # Top 5 entities
        generated_queries.append({"type": "entity", "query": entity})

    return PreprocessedQuery(
        original_query=query,
        search_query=query,  # Keep original for vector search
        strategy_used="graphrag",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        generated_queries=generated_queries,
        # Store entity list in metadata
        decomposition_response=f"Entities: {', '.join(query_entities)}" if query_entities else "",
    )


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

# Maps strategy ID to strategy function
STRATEGIES: Dict[str, StrategyFunction] = {
    "none": none_strategy,
    "hyde": hyde_strategy,
    "decomposition": decomposition_strategy,
    "graphrag": graphrag_strategy,
}


def get_strategy(strategy_id: str) -> StrategyFunction:
    """Get strategy function by ID.

    Args:
        strategy_id: One of "none", "hyde", "decomposition".

    Returns:
        Strategy function that takes (query, model) and returns PreprocessedQuery.

    Raises:
        ValueError: If strategy_id is not registered.

    Example:
        >>> strategy_fn = get_strategy("hyde")
        >>> result = strategy_fn("Why do we procrastinate?", model="openai/gpt-4o-mini")
        >>> result.strategy_used
        "hyde"
    """
    if strategy_id not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")
    return STRATEGIES[strategy_id]


def list_strategies() -> List[str]:
    """List all registered strategy IDs.

    Returns:
        List of strategy IDs (e.g., ["none", "hyde", "decomposition"]).
    """
    return list(STRATEGIES.keys())
