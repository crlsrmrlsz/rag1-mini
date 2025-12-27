"""Query preprocessing for improved RAG retrieval.

## RAG Theory: Query Preprocessing

Query preprocessing improves retrieval by transforming queries before searching.
Each preprocessing strategy applies its transformation directly to any query,
following the original research papers' design.

Techniques implemented:
1. **HyDE** generates hypothetical answers for semantic matching
   - Paper: "Precise Zero-Shot Dense Retrieval" (arXiv:2212.10496)
2. **Query decomposition** breaks complex questions into sub-queries
   - Research: Query Decomposition showed +36.7% MRR@10 improvement (arXiv:2507.00355)

## Library Usage

Uses OpenRouter API via `requests` for LLM calls.
Pydantic schemas ensure structured LLM outputs.

## Data Flow

1. User selects preprocessing strategy (none, hyde, decomposition)
2. Strategy transforms query directly (no classification routing)
3. Return PreprocessedQuery with original, transformed, and metadata
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import requests
from pydantic import ValidationError as PydanticValidationError

from src.config import PREPROCESSING_MODEL
from src.shared.files import setup_logging
from src.shared.openrouter_client import call_chat_completion, call_structured_completion
from src.rag_pipeline.retrieval.preprocessing.schemas import (
    DecompositionResult,
)

logger = setup_logging(__name__)


@dataclass
class PreprocessedQuery:
    """Result of query preprocessing.

    Attributes:
        original_query: The user's original query text.
        search_query: The query to use for retrieval (may be transformed).
        strategy_used: The preprocessing strategy that was applied.
        preprocessing_time_ms: Time taken for preprocessing in milliseconds.
        model: Model ID used for preprocessing (for logging).
        hyde_passage: The hypothetical passage generated (hyde strategy).
        sub_queries: Decomposed sub-questions (decomposition strategy).
        generated_queries: List of {type, query} dicts for multi-query retrieval.
        decomposition_response: Raw decomposition response from LLM.
    """

    original_query: str
    search_query: str
    strategy_used: str = ""
    preprocessing_time_ms: float = 0.0
    model: str = ""
    # HyDE strategy fields
    hyde_passage: Optional[str] = None
    # Decomposition strategy fields (generated_queries used for RRF)
    generated_queries: List[Dict[str, str]] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    decomposition_response: Optional[str] = None


# =============================================================================
# HyDE: HYPOTHETICAL DOCUMENT EMBEDDINGS
# =============================================================================

HYDE_PROMPT = """Please write a passage from a cognitive science and philosophy knowledge base to answer the question.

Question: {query}

Passage:"""


# =============================================================================
# DECOMPOSITION PROMPTS
# =============================================================================

DECOMPOSITION_PROMPT = """Break down this question for a knowledge base covering cognitive science and philosophical wisdom traditions.

Create 3-4 sub-questions that together would answer the original question.
Each sub-question should target a specific aspect that could be answered by a passage in the knowledge base.

Question: "{query}"

Respond with JSON:
{{
  "sub_questions": ["...", "...", "..."],
  "reasoning": "Brief explanation"
}}"""


def hyde_prompt(query: str, model: Optional[str] = None) -> str:
    """Generate hypothetical answer for HyDE retrieval.

    HyDE (Hypothetical Document Embeddings) generates a plausible answer
    to the query, then searches for real passages similar to this answer.
    This bridges the semantic gap between questions and document passages.

    Paper: arXiv:2212.10496 - "Precise Zero-Shot Dense Retrieval without
    Relevance Labels"

    Args:
        query: The user's original question.
        model: Override model (defaults to PREPROCESSING_MODEL).

    Returns:
        A hypothetical passage that would answer the query.

    Example:
        >>> hyde_prompt("Why do we procrastinate?")
        "Procrastination stems from temporal discounting..."
    """
    model = model or PREPROCESSING_MODEL

    # Format prompt with query only (self-contained domain description)
    prompt = HYDE_PROMPT.format(query=query)

    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_chat_completion(
            messages=messages,
            model=model,
            temperature=0.7,  # Paper uses 0.7 for diverse hypothetical documents
            max_tokens=300,   # Allow natural length; encoder filters noise (paper approach)
        )

        return response.strip()

    except requests.RequestException as e:
        logger.warning(f"HyDE prompt failed: {e}, using original query")
        return query


# =============================================================================
# QUERY DECOMPOSITION
# =============================================================================


def decompose_query(query: str, model: Optional[str] = None) -> Tuple[List[str], str]:
    """Decompose a complex query into sub-questions for parallel retrieval.

    Breaks complex comparison or multi-aspect questions into simpler
    sub-questions that can be answered independently. Each sub-question
    is used for retrieval, with results merged using RRF.

    Uses Pydantic DecompositionResult schema for type-safe extraction.
    The schema guarantees sub_questions is a List[str].

    Args:
        query: The user's original query.
        model: Override model (defaults to PREPROCESSING_MODEL).

    Returns:
        Tuple of (list of sub-questions, raw LLM response string).

    Example:
        >>> sub_qs, resp = decompose_query("Compare Stoic and Buddhist views on suffering")
        >>> len(sub_qs)
        3
        >>> sub_qs[0]
        "What is the Stoic view on suffering and how to overcome it?"
    """
    model = model or PREPROCESSING_MODEL

    # Format prompt with query only (self-contained domain description)
    prompt = DECOMPOSITION_PROMPT.format(query=query)

    messages = [
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_structured_completion(
            messages=messages,
            model=model,
            response_model=DecompositionResult,
            temperature=0.3,  # Slight creativity for varied sub-questions
            max_tokens=400,
        )

        # Pydantic guarantees sub_questions is List[str]
        if not result.sub_questions:
            logger.warning("[decompose] No sub-questions extracted, using original")
            return [query], result.model_dump_json()

        logger.info(f"[decompose] Generated {len(result.sub_questions)} sub-questions")
        return result.sub_questions, result.model_dump_json()

    except (PydanticValidationError, Exception) as e:
        logger.warning(f"[decompose] Error: {e}, using original query")
        return [query], str(e)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def preprocess_query(
    query: str,
    model: Optional[str] = None,
    strategy: Optional[str] = None,
) -> PreprocessedQuery:
    """Preprocess a query for optimal retrieval.

    Main entry point for query preprocessing. Dispatches to the appropriate
    strategy based on the strategy parameter. Each strategy applies its
    transformation directly to the query without classification.

    Args:
        query: The user's original query.
        model: Override model for LLM calls.
        strategy: Preprocessing strategy ID. Options:
            - "none": Return original query unchanged (no LLM calls)
            - "hyde": Generate hypothetical answer for semantic matching
            - "decomposition": Break into sub-questions + RRF merge

    Returns:
        PreprocessedQuery with transformed query and strategy_used.

    Example:
        >>> result = preprocess_query("How should I live?", strategy="hyde")
        >>> result.search_query
        "Eudaimonia, the ancient Greek concept of flourishing..."
        >>> result.strategy_used
        "hyde"
    """
    # Import here to avoid circular imports
    from src.config import DEFAULT_PREPROCESSING_STRATEGY
    from src.rag_pipeline.retrieval.preprocessing.strategies import get_strategy

    # Use default strategy if none specified
    if strategy is None:
        strategy = DEFAULT_PREPROCESSING_STRATEGY

    # Get and execute the strategy function
    strategy_fn = get_strategy(strategy)
    return strategy_fn(query, model=model)
