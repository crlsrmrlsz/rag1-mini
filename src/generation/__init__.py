"""Answer generation module for RAG1-Mini.

Provides LLM-based answer synthesis from retrieved chunks:
- Query-type-aware prompting (FACTUAL, OPEN_ENDED, MULTI_HOP)
- Source citation and attribution
- Structured answer output
"""

from src.generation.answer_generator import (
    GeneratedAnswer,
    generate_answer,
)

__all__ = [
    "GeneratedAnswer",
    "generate_answer",
]
