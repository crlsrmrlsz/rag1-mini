"""Preprocessing strategy implementations.

## RAG Theory: Strategy Pattern for Query Preprocessing

Each strategy applies its transformation directly to any query, following
the original research papers' design. The strategies don't use classification
routing - they simply transform queries for better retrieval.

Strategies:
- none: No transformation (baseline for comparison)
- step_back: Always abstract to broader concepts (+27% on multi-hop, arXiv:2310.06117)
- decomposition: Always break into sub-questions + RRF merge (+36.7% MRR@10)

The strategy pattern allows easy A/B testing and adding new strategies
without modifying existing code.

## Library Usage

Uses the existing query_preprocessing functions (step_back_prompt, etc.)
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
    step_back_prompt as _step_back_prompt_fn,
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


def step_back_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Always apply step-back prompting to broaden the query.

    Transforms any query into broader concepts using vocabulary that matches
    the knowledge base, improving retrieval coverage.

    Based on "Take a Step Back" (Google DeepMind, 2023) which showed
    +27% improvement on multi-hop reasoning tasks.

    Args:
        query: The user's original query.
        model: Model for step-back LLM call.

    Returns:
        PreprocessedQuery with transformed search_query.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Apply step-back transformation
    step_back_query = _step_back_prompt_fn(query, model=model)
    logger.info(f"[step_back] Transformed: {step_back_query[:80]}...")

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        search_query=step_back_query,
        step_back_query=step_back_query,
        step_back_response=step_back_query,
        strategy_used="step_back",
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
# STRATEGY REGISTRY
# =============================================================================

# Maps strategy ID to strategy function
STRATEGIES: Dict[str, StrategyFunction] = {
    "none": none_strategy,
    "step_back": step_back_strategy,
    "decomposition": decomposition_strategy,
}


def get_strategy(strategy_id: str) -> StrategyFunction:
    """Get strategy function by ID.

    Args:
        strategy_id: One of "none", "step_back", "decomposition".

    Returns:
        Strategy function that takes (query, model) and returns PreprocessedQuery.

    Raises:
        ValueError: If strategy_id is not registered.

    Example:
        >>> strategy_fn = get_strategy("decomposition")
        >>> result = strategy_fn("Compare Stoic and Buddhist views", model="openai/gpt-4o-mini")
        >>> result.strategy_used
        "decomposition"
    """
    if strategy_id not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")
    return STRATEGIES[strategy_id]


def list_strategies() -> List[str]:
    """List all registered strategy IDs.

    Returns:
        List of strategy IDs (e.g., ["none", "step_back", "decomposition"]).
    """
    return list(STRATEGIES.keys())
