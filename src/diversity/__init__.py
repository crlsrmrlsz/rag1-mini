"""Diversity balancing for RAG retrieval results.

Ensures search results include representation from multiple source categories
(neuroscience and philosophy) to provide balanced perspectives.
"""

from src.diversity.source_balancer import (
    DiversityResult,
    apply_diversity_balance,
    get_book_category,
)

__all__ = ["apply_diversity_balance", "get_book_category", "DiversityResult"]
