"""Preprocessing strategy implementations.

## RAG Theory: Strategy Pattern for Query Preprocessing

Different query types benefit from different preprocessing approaches:
- Factual queries: Often work best with the original query
- Open-ended queries: Benefit from step-back prompting (broader concepts)
- Multi-hop queries: Will benefit from decomposition (future)

The strategy pattern allows easy switching between approaches for A/B testing
and enables adding new strategies without modifying existing code.

## Library Usage

Uses the existing query_classifier functions (classify_query, step_back_prompt)
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
from src.preprocessing.query_classifier import (
    CLASSIFICATION_PROMPT,
    STEP_BACK_PROMPT,
    PreprocessedQuery,
    QueryType,
    classify_query,
    step_back_prompt as _step_back_prompt_fn,
)
from src.utils.file_utils import setup_logging

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
        query_type=QueryType.FACTUAL,  # Default type (no classification)
        search_query=query,
        strategy_used="none",
        preprocessing_time_ms=0.0,
        model=model or PREPROCESSING_MODEL,
    )


def baseline_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Classify query type only, no query transformation.

    Use case: Track query type distribution without modifying search queries.
    Enables query-type-aware answer generation prompts while keeping
    original query for retrieval.

    Args:
        query: The user's original query.
        model: Model for classification LLM call.

    Returns:
        PreprocessedQuery with classification but search_query unchanged.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    query_type, classification_response = classify_query(query, model=model)
    logger.info(f"[baseline] Query classified as: {query_type.value}")

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        query_type=query_type,
        search_query=query,  # Unchanged
        strategy_used="baseline",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        classification_prompt_used=CLASSIFICATION_PROMPT,
        classification_response=classification_response,
    )


def step_back_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Classify + apply step-back prompting for open-ended queries.

    This is the current default implementation. For OPEN_ENDED queries,
    generates a broader search query using step-back prompting with
    domain-specific vocabulary.

    Based on "Take a Step Back" (Google DeepMind, 2023) which showed
    +27% improvement on multi-hop reasoning tasks.

    Args:
        query: The user's original query.
        model: Model for LLM calls (classification + step-back).

    Returns:
        PreprocessedQuery with transformed search_query for OPEN_ENDED.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Step 1: Classify the query
    query_type, classification_response = classify_query(query, model=model)
    logger.info(f"[step_back] Query classified as: {query_type.value}")

    # Step 2: Transform if open-ended
    step_back_query = None
    search_query = query
    step_back_prompt_used = None
    step_back_response = None

    if query_type == QueryType.OPEN_ENDED:
        step_back_query = _step_back_prompt_fn(query, model=model)
        step_back_response = step_back_query
        search_query = step_back_query
        step_back_prompt_used = STEP_BACK_PROMPT
        logger.info(f"[step_back] Transformed query: {step_back_query}")

    elif query_type == QueryType.MULTI_HOP:
        # Future: Implement query decomposition
        logger.info("[step_back] Multi-hop detected (decomposition not yet implemented)")

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        query_type=query_type,
        search_query=search_query,
        step_back_query=step_back_query,
        strategy_used="step_back",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        classification_prompt_used=CLASSIFICATION_PROMPT,
        step_back_prompt_used=step_back_prompt_used,
        classification_response=classification_response,
        step_back_response=step_back_response,
    )


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

# Maps strategy ID to strategy function
STRATEGIES: Dict[str, StrategyFunction] = {
    "none": none_strategy,
    "baseline": baseline_strategy,
    "step_back": step_back_strategy,
}


def get_strategy(strategy_id: str) -> StrategyFunction:
    """Get strategy function by ID.

    Args:
        strategy_id: One of "none", "baseline", "step_back".

    Returns:
        Strategy function that takes (query, model) and returns PreprocessedQuery.

    Raises:
        ValueError: If strategy_id is not registered.

    Example:
        >>> strategy_fn = get_strategy("step_back")
        >>> result = strategy_fn("How should I live?", model="openai/gpt-4o-mini")
        >>> result.strategy_used
        "step_back"
    """
    if strategy_id not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")
    return STRATEGIES[strategy_id]


def list_strategies() -> List[str]:
    """List all registered strategy IDs.

    Returns:
        List of strategy IDs (e.g., ["none", "baseline", "step_back"]).
    """
    return list(STRATEGIES.keys())


# =============================================================================
# FUTURE STRATEGY STUBS
# =============================================================================

# TODO (Phase 3): Implement multi_query_strategy
# - Generate 3-4 targeted queries using principle extraction
# - Retrieve for each query
# - Merge results using Reciprocal Rank Fusion (RRF)
# - Expected improvement: Better coverage for complex questions
#
# def multi_query_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
#     """Multi-query with RRF merging."""
#     pass

# TODO (Phase 4): Implement decomposition_strategy
# - Detect MULTI_HOP queries (comparisons, multi-aspect questions)
# - Break into sub-questions
# - Retrieve for each sub-question
# - Merge results with RRF
# - Expected improvement: +36.7% MRR@10 per research
#
# def decomposition_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
#     """Query decomposition for MULTI_HOP questions."""
#     pass
