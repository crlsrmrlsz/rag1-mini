"""RAG evaluation module using RAGAS framework.

Provides:
- RAGAS metric evaluation for RAG responses
- Test question management
- Evaluation report generation
"""

from src.evaluation.ragas_evaluator import (
    run_evaluation,
    create_evaluator_llm,
    retrieve_contexts,
    generate_answer,
)

__all__ = [
    "run_evaluation",
    "create_evaluator_llm",
    "retrieve_contexts",
    "generate_answer",
]
