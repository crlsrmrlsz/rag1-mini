"""Stage 4.5: Auto-Tune Entity Types for GraphRAG.

## RAG Theory: Auto-Tuning for Domain-Specific Extraction

Instead of manually defining entity types, this stage discovers them from
the actual corpus content. Benefits:

1. Entity types match what's actually in the text
2. Query-time extraction uses same vocabulary as indexed entities
3. No manual prompt engineering required

## Process

1. Open-ended extraction on all chunks (LLM assigns types freely)
2. Aggregate discovered types with counts
3. LLM consolidates into clean taxonomy (15-25 entity types, 10-20 relationships)
4. Save to discovered_types.json for use in extraction and queries

## Usage

```bash
# Full auto-tuning (all books, resumable)
python -m src.stages.run_stage_4_5_autotune

# Resume after interruption (skip already processed books)
python -m src.stages.run_stage_4_5_autotune --overwrite skip

# Force reprocess all books
python -m src.stages.run_stage_4_5_autotune --overwrite all

# Skip consolidation (just extraction)
python -m src.stages.run_stage_4_5_autotune --skip-consolidation

# Show previously discovered types
python -m src.stages.run_stage_4_5_autotune --show-types
```

## Output

- data/processed/05_final_chunks/graph/extractions/{book}.json (per-book)
- data/processed/05_final_chunks/graph/extraction_results.json (merged)
- data/processed/05_final_chunks/graph/discovered_types.json (taxonomy)
- data/logs/autotune_TIMESTAMP.log (execution log)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.graph.auto_tuning import (
    run_auto_tuning_resumable,
    load_discovered_types,
    load_book_files,
)
from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg
from src.config import GRAPHRAG_EXTRACTION_MODEL, DIR_CLEANING_LOGS

logger = setup_logging(__name__)


def setup_file_logging() -> Path:
    """Setup dual console + file logging.

    Creates a timestamped log file that captures all logging output
    alongside the console output.

    Returns:
        Path to the created log file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = DIR_CLEANING_LOGS / f"autotune_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Add FileHandler to root logger
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(fh)

    return log_file


def main():
    """Run auto-tuning stage with per-book resume support."""
    parser = argparse.ArgumentParser(
        description="Stage 4.5: Auto-tune entity types for GraphRAG"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="section",
        help="Chunking strategy (default: section)",
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        choices=["prompt", "skip", "all"],
        default="prompt",
        help="Overwrite mode: prompt (default), skip existing books, or overwrite all",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GRAPHRAG_EXTRACTION_MODEL,
        help=f"LLM model (default: {GRAPHRAG_EXTRACTION_MODEL})",
    )
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="Skip LLM consolidation step",
    )
    parser.add_argument(
        "--show-types",
        action="store_true",
        help="Show previously discovered types and exit",
    )
    parser.add_argument(
        "--list-books",
        action="store_true",
        help="List books to be processed and exit",
    )

    args = parser.parse_args()

    # Show existing types if requested
    if args.show_types:
        try:
            types = load_discovered_types()
            print("\n=== Discovered Entity Types ===")
            for t in types["entity_types"]:
                print(f"  - {t}")
            print(f"\n=== Discovered Relationship Types ===")
            for t in types["relationship_types"]:
                print(f"  - {t}")
            return
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    # List books if requested
    if args.list_books:
        try:
            book_files = load_book_files(args.strategy)
            print(f"\n=== Books to Process ({len(book_files)}) ===")
            for i, book_path in enumerate(book_files, 1):
                print(f"  {i:2}. {book_path.stem}")
            return
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    # Setup file logging
    log_file = setup_file_logging()

    # Create overwrite context
    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    # Run auto-tuning
    logger.info("=" * 60)
    logger.info("Stage 4.5: Auto-Tune Entity Types (Resumable)")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Overwrite mode: {args.overwrite}")
    logger.info(f"Model: {args.model}")

    try:
        results = run_auto_tuning_resumable(
            strategy=args.strategy,
            overwrite_context=overwrite_context,
            model=args.model,
            skip_consolidation=args.skip_consolidation,
        )
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Interrupted by user")
        logger.info("=" * 60)
        logger.info("Partial work has been saved. Resume with: --overwrite skip")
        logger.info(f"Log file: {log_file}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Auto-tuning failed: {e}")
        logger.info("Partial work has been saved. Resume with: --overwrite skip")
        logger.info(f"Log file: {log_file}")
        raise

    # Summary
    logger.info("=" * 60)
    logger.info("Auto-Tuning Complete")
    logger.info("=" * 60)
    logger.info(f"Books processed: {len(results.get('processed_books', []))}")
    logger.info(f"Books skipped: {len(results.get('skipped_books', []))}")
    logger.info(f"Total chunks: {results['stats']['processed_chunks']}")
    logger.info(f"Total entities: {results['stats']['total_entities']}")
    logger.info(f"Total relationships: {results['stats']['total_relationships']}")
    logger.info(f"Unique entity types: {results['stats'].get('unique_entity_types', 'N/A')}")
    logger.info(f"Unique relationship types: {results['stats'].get('unique_relationship_types', 'N/A')}")

    if "consolidated_types" in results:
        logger.info("")
        logger.info("Consolidated Entity Types:")
        for t in results["consolidated_types"]["entity_types"]:
            logger.info(f"  - {t}")
        logger.info("")
        logger.info("Consolidated Relationship Types:")
        for t in results["consolidated_types"]["relationship_types"]:
            logger.info(f"  - {t}")

    logger.info("")
    logger.info(f"Per-book extractions: {results.get('extractions_dir', 'N/A')}")
    logger.info(f"Merged results: {results['extraction_path']}")
    if "types_path" in results:
        logger.info(f"Discovered types: {results['types_path']}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
