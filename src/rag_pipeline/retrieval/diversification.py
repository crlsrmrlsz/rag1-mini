"""Source-type diversity balancing for RAG results.

This module ensures search results include balanced representation from
multiple source categories (neuroscience and philosophy) to provide
diverse perspectives in answers.

## Theory: Diversity in RAG

RAG systems can exhibit source bias when one category dominates the embedding
space or has more chunks. This module implements post-retrieval rebalancing
to ensure both philosophical and scientific perspectives are represented,
which is essential for a "hybrid neuroscientist + philosopher" AI.

## Library: dataclasses

Uses Python's built-in dataclasses for clean result containers. No external
dependencies needed since this is simple filtering and sorting logic.

## Data Flow

1. Receives scored SearchResult list (after reranking)
2. Partitions by category using BOOK_CATEGORIES lookup
3. Filters by minimum score threshold
4. Takes proportional amounts from each category
5. Returns balanced, sorted results with metadata
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import BOOK_CATEGORIES, DIVERSITY_BALANCE, DIVERSITY_MIN_SCORE
from src.shared.files import setup_logging

logger = setup_logging(__name__)


@dataclass
class DiversityResult:
    """Result of diversity balancing with logging metadata.

    Attributes:
        results: Balanced list of SearchResult objects.
        category_counts: Actual counts per category.
        excluded_count: Number of results excluded due to min_score threshold.
        original_count: Number of results before balancing.
    """

    results: List[Any]  # List[SearchResult] - avoid circular import
    category_counts: Dict[str, int]
    excluded_count: int
    original_count: int


def get_book_category(book_id: str) -> Optional[str]:
    """Look up the category for a book_id using fuzzy matching.

    Args:
        book_id: Book identifier from chunk metadata.

    Returns:
        Category string ("neuroscience" or "philosophy") or None if unknown.

    Note:
        Uses BOOK_CATEGORIES from config for runtime lookup.
        Matching is fuzzy to handle slight naming variations.
    """
    for category, books in BOOK_CATEGORIES.items():
        for book in books:
            # Fuzzy match: check if either contains the other
            if book_id in book or book in book_id:
                return category
    return None


def apply_diversity_balance(
    results: List[Any],
    target_count: int,
    balance: Optional[Dict[str, float]] = None,
    min_score: Optional[float] = None,
) -> DiversityResult:
    """Balance search results to ensure source-type diversity.

    This function redistributes results to match target category ratios
    while respecting a minimum score threshold. It operates AFTER reranking
    when scores are most accurate.

    Args:
        results: Scored SearchResult objects (sorted by score descending).
        target_count: Desired number of final results.
        balance: Category weights (e.g., {"neuroscience": 0.6, "philosophy": 0.4}).
                 Defaults to DIVERSITY_BALANCE from config.
        min_score: Minimum score threshold. Results below this are excluded
                   even if category quota is not met.

    Returns:
        DiversityResult containing balanced results and metadata for logging.

    Algorithm:
        1. Partition results by category
        2. Filter each partition by min_score
        3. Calculate target slots per category (target_count * balance[cat])
        4. Take up to target slots from each category (preserving score order)
        5. If a category has fewer results than its quota, do NOT fill from other
        6. Sort combined results by score to maintain relevance order

    Example:
        >>> results = [...]  # 10 results after reranking
        >>> balanced = apply_diversity_balance(results, target_count=10)
        >>> # Returns ~6 neuroscience, ~4 philosophy (if both have enough)
        >>> # If philosophy only has 2 above min_score, returns 6 + 2 = 8 total
    """
    balance = balance or DIVERSITY_BALANCE
    min_score = min_score if min_score is not None else DIVERSITY_MIN_SCORE
    original_count = len(results)

    # Partition by category
    by_category: Dict[str, List[Any]] = {
        "neuroscience": [],
        "philosophy": [],
        "other": [],
    }

    for r in results:
        cat = get_book_category(r.book_id) or "other"
        by_category[cat].append(r)

    # Debug: log sample book_ids to diagnose categorization
    if results:
        sample_ids = [r.book_id for r in results[:3]]
        logger.info(f"Diversity sample book_ids: {sample_ids}")

    logger.info(
        "Diversity partition: neuro=%d, phil=%d, other=%d",
        len(by_category["neuroscience"]),
        len(by_category["philosophy"]),
        len(by_category["other"]),
    )

    # Filter by min_score
    excluded = 0
    for cat in by_category:
        before = len(by_category[cat])
        by_category[cat] = [r for r in by_category[cat] if r.score >= min_score]
        excluded += before - len(by_category[cat])

    if excluded > 0:
        logger.debug("Diversity filter: excluded %d results below min_score=%.2f", excluded, min_score)

    # Calculate targets
    neuro_target = round(target_count * balance.get("neuroscience", 0.5))
    phil_target = target_count - neuro_target

    # Take from each category (up to target, preserving score order)
    neuro_take = by_category["neuroscience"][:neuro_target]
    phil_take = by_category["philosophy"][:phil_target]

    # Combine and sort by score (highest first)
    combined = neuro_take + phil_take
    combined.sort(key=lambda r: r.score, reverse=True)

    category_counts = {
        "neuroscience": len(neuro_take),
        "philosophy": len(phil_take),
    }

    logger.info(
        "Diversity balance: %d neuro + %d phil = %d total (from %d, excluded %d)",
        category_counts["neuroscience"],
        category_counts["philosophy"],
        len(combined),
        original_count,
        excluded,
    )

    return DiversityResult(
        results=combined,
        category_counts=category_counts,
        excluded_count=excluded,
        original_count=original_count,
    )
