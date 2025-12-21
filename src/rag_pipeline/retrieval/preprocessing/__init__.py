"""Query preprocessing module for RAG1-Mini.

Provides query classification and transformation for improved retrieval:
- Query type classification (FACTUAL, OPEN_ENDED, MULTI_HOP)
- Step-back prompting for open-ended queries
- Strategy-based preprocessing with registry pattern
- Query decomposition for multi-hop questions (planned)
"""

from src.rag_pipeline.retrieval.preprocessing.query_classifier import (
    QueryType,
    PreprocessedQuery,
    classify_query,
    step_back_prompt,
    preprocess_query,
)
from src.rag_pipeline.retrieval.preprocessing.strategies import (
    get_strategy,
    list_strategies,
    STRATEGIES,
)

__all__ = [
    # Core types
    "QueryType",
    "PreprocessedQuery",
    # Main entry point
    "preprocess_query",
    # Low-level functions (for direct use)
    "classify_query",
    "step_back_prompt",
    # Strategy registry
    "get_strategy",
    "list_strategies",
    "STRATEGIES",
]
