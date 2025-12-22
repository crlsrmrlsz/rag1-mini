"""Query preprocessing module for RAG1-Mini.

Provides query transformation strategies for improved retrieval:
- Step-back prompting to abstract to broader concepts
- Multi-query generation for diverse retrieval
- Query decomposition for complex questions
- Strategy-based preprocessing with registry pattern
"""

from src.rag_pipeline.retrieval.preprocessing.query_preprocessing import (
    PreprocessedQuery,
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
    "PreprocessedQuery",
    # Main entry point
    "preprocess_query",
    # Low-level functions (for direct use)
    "step_back_prompt",
    # Strategy registry
    "get_strategy",
    "list_strategies",
    "STRATEGIES",
]
