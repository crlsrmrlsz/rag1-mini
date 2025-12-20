"""Query preprocessing module for RAG1-Mini.

Provides query classification and transformation for improved retrieval:
- Query type classification (FACTUAL, OPEN_ENDED, MULTI_HOP)
- Step-back prompting for open-ended queries
- Query decomposition for multi-hop questions
"""

from src.preprocessing.query_classifier import (
    QueryType,
    PreprocessedQuery,
    classify_query,
    step_back_prompt,
    preprocess_query,
)

__all__ = [
    "QueryType",
    "PreprocessedQuery",
    "classify_query",
    "step_back_prompt",
    "preprocess_query",
]
