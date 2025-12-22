"""Preprocessing strategy implementations.

## RAG Theory: Strategy Pattern for Query Preprocessing

Different query types benefit from different preprocessing approaches:
- Factual queries: Often work best with the original query
- Open-ended queries: Benefit from step-back prompting (broader concepts)
- Multi-hop queries: Will benefit from decomposition (future)

The strategy pattern allows easy switching between approaches for A/B testing
and enables adding new strategies without modifying existing code.

## Library Usage

Uses the existing query_preprocessing functions (classify_query, step_back_prompt)
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
    CLASSIFICATION_PROMPT,
    STEP_BACK_PROMPT,
    PRINCIPLE_EXTRACTION_PROMPT,
    MULTI_QUERY_PROMPT,
    DECOMPOSITION_PROMPT,
    PreprocessedQuery,
    QueryType,
    classify_query,
    step_back_prompt as _step_back_prompt_fn,
    extract_principles,
    generate_multi_queries,
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
# MULTI-QUERY STRATEGY
# =============================================================================


def multi_query_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Generate multiple targeted queries for RRF merging.

    This strategy implements multi-query retrieval:
    1. Classifies the query type
    2. Extracts underlying principles/concepts
    3. Generates 4 targeted queries (neuroscience, philosophy, bridging, broad)
    4. Includes original query for exact-match coverage

    Based on Multi-Query RAG research showing improved coverage for
    multi-faceted questions. Results are merged using Reciprocal Rank
    Fusion (RRF) in the retrieval stage.

    Args:
        query: The user's original query.
        model: Model for LLM calls.

    Returns:
        PreprocessedQuery with generated_queries populated for RRF merging.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Step 1: Classify the query
    query_type, classification_response = classify_query(query, model=model)
    logger.info(f"[multi_query] Query classified as: {query_type.value}")

    # Step 2: Extract principles
    principles = extract_principles(query, model=model)
    logger.info(f"[multi_query] Extracted core topic: {principles.get('core_topic', 'N/A')}")

    # Step 3: Generate multiple queries
    generated = generate_multi_queries(query, principles, model=model)
    logger.info(f"[multi_query] Generated {len(generated)} queries")

    # Primary search query is the "broad" query or first available
    primary_query = query  # Default to original
    for q in generated:
        if q.get("type") == "broad":
            primary_query = q.get("query", query)
            break
    if primary_query == query and generated:
        primary_query = generated[0].get("query", query)

    # Add original query to the mix (first in list for RRF priority)
    all_queries = [{"type": "original", "query": query}] + generated

    # Build the prompts used for logging
    principle_prompt = PRINCIPLE_EXTRACTION_PROMPT.format(query=query)
    multi_query_prompt = MULTI_QUERY_PROMPT.format(
        query=query,
        core_topic=principles.get("core_topic", ""),
        neuro_concepts=", ".join(principles.get("neuroscience_concepts", [])),
        philo_concepts=", ".join(principles.get("philosophical_concepts", [])),
        related_terms=", ".join(principles.get("related_terms", [])),
    )

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        query_type=query_type,
        search_query=primary_query,
        strategy_used="multi_query",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        classification_prompt_used=CLASSIFICATION_PROMPT,
        classification_response=classification_response,
        generated_queries=all_queries,
        principle_extraction=principles,
        principle_extraction_prompt_used=principle_prompt,
        multi_query_prompt_used=multi_query_prompt,
    )


# =============================================================================
# DECOMPOSITION STRATEGY
# =============================================================================


def decomposition_strategy(query: str, model: Optional[str] = None) -> PreprocessedQuery:
    """Decompose MULTI_HOP queries into sub-questions for RRF merging.

    This strategy specifically handles comparison and multi-aspect questions:
    1. Classifies the query type
    2. If MULTI_HOP, decomposes into 2-4 sub-questions
    3. Sub-questions are used for RRF-merged retrieval

    Based on Query Decomposition research showing +36.7% MRR@10 improvement
    for complex multi-hop queries.

    Args:
        query: The user's original query.
        model: Model for LLM calls.

    Returns:
        PreprocessedQuery with sub_queries populated for MULTI_HOP.
    """
    start_time = time.time()
    model = model or PREPROCESSING_MODEL

    # Step 1: Classify the query
    query_type, classification_response = classify_query(query, model=model)
    logger.info(f"[decomposition] Query classified as: {query_type.value}")

    # Step 2: Decompose if MULTI_HOP
    sub_queries = []
    decomposition_response = None
    decomposition_prompt_used = None

    if query_type == QueryType.MULTI_HOP:
        sub_queries, decomposition_response = decompose_query(query, model=model)
        decomposition_prompt_used = DECOMPOSITION_PROMPT.format(query=query)
        logger.info(f"[decomposition] Decomposed into {len(sub_queries)} sub-queries")
    else:
        # For non-MULTI_HOP, just use original query
        logger.info("[decomposition] Not MULTI_HOP, using original query")

    # Build generated_queries format for search compatibility
    # This allows reuse of existing RRF merging infrastructure
    generated_queries = [{"type": "original", "query": query}]
    for i, sq in enumerate(sub_queries):
        generated_queries.append({"type": f"sub_{i+1}", "query": sq})

    elapsed_ms = (time.time() - start_time) * 1000

    return PreprocessedQuery(
        original_query=query,
        query_type=query_type,
        search_query=query,  # Keep original for display
        sub_queries=sub_queries,
        strategy_used="decomposition",
        preprocessing_time_ms=elapsed_ms,
        model=model,
        classification_prompt_used=CLASSIFICATION_PROMPT,
        classification_response=classification_response,
        generated_queries=generated_queries,  # For search/RRF compatibility
        decomposition_prompt_used=decomposition_prompt_used,
        decomposition_response=decomposition_response,
    )


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

# Maps strategy ID to strategy function
STRATEGIES: Dict[str, StrategyFunction] = {
    "none": none_strategy,
    "baseline": baseline_strategy,
    "step_back": step_back_strategy,
    "multi_query": multi_query_strategy,
    "decomposition": decomposition_strategy,
}


def get_strategy(strategy_id: str) -> StrategyFunction:
    """Get strategy function by ID.

    Args:
        strategy_id: One of "none", "baseline", "step_back", "multi_query", "decomposition".

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
        List of strategy IDs (e.g., ["none", "baseline", "step_back", "multi_query", "decomposition"]).
    """
    return list(STRATEGIES.keys())
