"""Stage 4: Chunking with strategy selection.

Supports multiple chunking strategies:
- section: Sequential with overlap (baseline, fast)
- semantic: Embedding similarity-based (better coherence, uses API)
- contextual: LLM-generated chunk context (future, Anthropic-style)
- raptor: Hierarchical summarization tree (future)

Usage:
    python -m src.stages.run_stage_4_chunking                    # Default: section
    python -m src.stages.run_stage_4_chunking --strategy semantic  # Semantic chunking
"""

import argparse
from pathlib import Path

from src.config import (
    DIR_NLP_CHUNKS,
    DIR_FINAL_CHUNKS,
    DEFAULT_CHUNKING_STRATEGY,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from src.shared import setup_logging, get_file_list
from src.rag_pipeline.chunking.strategies import get_strategy, list_strategies

logger = setup_logging("Stage4_Chunking")


def main():
    """Run chunking with selected strategy."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Chunking with strategy selection"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=list_strategies(),
        help=f"Chunking strategy (default: {DEFAULT_CHUNKING_STRATEGY})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Semantic similarity threshold (0.0-1.0). "
            f"Lower = fewer splits (larger chunks). "
            f"Only used with semantic strategy. (default: {SEMANTIC_SIMILARITY_THRESHOLD})"
        ),
    )
    args = parser.parse_args()

    # Build strategy kwargs
    strategy_kwargs = {}
    if args.threshold is not None:
        if args.strategy != "semantic":
            logger.warning("--threshold is only used with semantic strategy, ignoring")
        else:
            strategy_kwargs["similarity_threshold"] = args.threshold

    logger.info(f"Starting Stage 4: Chunking (strategy: {args.strategy})")

    # Check Stage 3 output exists
    nlp_chunk_files = get_file_list(DIR_NLP_CHUNKS, "json")
    logger.info(f"Found {len(nlp_chunk_files)} NLP chunk files from Stage 3.")

    if not nlp_chunk_files:
        logger.warning(f"No NLP chunk files found in {DIR_NLP_CHUNKS}. Run Stage 3 first.")
        return

    # Get strategy function and run
    strategy_fn = get_strategy(args.strategy, **strategy_kwargs)
    stats = strategy_fn()

    # Verify output
    strategy_dir = DIR_FINAL_CHUNKS / args.strategy
    strategy_files = list(strategy_dir.glob("*.json")) if strategy_dir.exists() else []

    logger.info(f"Stage 4 complete ({args.strategy}). {len(strategy_files)} files created.")
    logger.info(f"Total chunks: {sum(stats.values())}")
    logger.info(f"Output: {strategy_dir}")


if __name__ == "__main__":
    main()
