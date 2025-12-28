"""Stage 4.5: Auto-Tune Entity Types for GraphRAG.

## IMPORTANT: This is an ALTERNATIVE to Stage 4.6 graph_extract

Both stages produce `extraction_results.json` - choose ONE:

| Stage | When to Use |
|-------|-------------|
| **4.5 autotune** | Recommended. Discovers entity types FROM your corpus. |
| **4.6 extract** | Uses predefined types from config.py. Faster but less adaptive. |

Do NOT run both - the second will overwrite the first.

## Usage

```bash
# Full auto-tuning (stratified consolidation by default)
python -m src.stages.run_stage_4_5_autotune

# Use global consolidation instead (original algorithm)
python -m src.stages.run_stage_4_5_autotune --consolidation global

# Resume after interruption
python -m src.stages.run_stage_4_5_autotune --overwrite skip

# Re-consolidate only (no re-extraction)
python -m src.stages.run_stage_4_5_autotune --reconsolidate stratified

# Show discovered types
python -m src.stages.run_stage_4_5_autotune --show-types
```

## Consolidation Strategies

- **stratified** (default): Balances entity types across corpora (neuroscience vs philosophy)
- **global**: Original algorithm, ranks by total count (larger corpora dominate)

## Next Step

After extraction, upload to Neo4j:
```bash
python -m src.stages.run_stage_6b_neo4j
```
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.graph.auto_tuning import (
    run_auto_tuning,
    load_discovered_types,
    load_book_files,
    reconsolidate,
)
from src.shared.files import setup_logging, OverwriteContext, parse_overwrite_arg
from src.config import GRAPHRAG_EXTRACTION_MODEL, DIR_CLEANING_LOGS

logger = setup_logging(__name__)


def setup_file_logging() -> Path:
    """Setup dual console + file logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = DIR_CLEANING_LOGS / f"autotune_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(fh)

    return log_file


def main():
    """Run auto-tuning stage."""
    parser = argparse.ArgumentParser(
        description="Stage 4.5: Auto-tune entity types for GraphRAG"
    )
    parser.add_argument(
        "--strategy", type=str, default="section",
        help="Chunking strategy (default: section)",
    )
    parser.add_argument(
        "--consolidation", type=str, default="stratified",
        choices=["stratified", "global"],
        help="Consolidation strategy: stratified (default) or global",
    )
    parser.add_argument(
        "--overwrite", type=str, default="prompt",
        choices=["prompt", "skip", "all"],
        help="Overwrite mode: prompt (default), skip existing, or overwrite all",
    )
    parser.add_argument(
        "--model", type=str, default=GRAPHRAG_EXTRACTION_MODEL,
        help=f"LLM model (default: {GRAPHRAG_EXTRACTION_MODEL})",
    )
    parser.add_argument(
        "--skip-consolidation", action="store_true",
        help="Skip LLM consolidation step",
    )
    parser.add_argument(
        "--show-types", action="store_true",
        help="Show previously discovered types and exit",
    )
    parser.add_argument(
        "--list-books", action="store_true",
        help="List books to be processed and exit",
    )
    parser.add_argument(
        "--reconsolidate", type=str, choices=["stratified", "global"],
        help="Re-run consolidation on existing extractions (no re-extraction)",
    )

    args = parser.parse_args()

    # Show existing types
    if args.show_types:
        try:
            types = load_discovered_types()
            print("\n=== Entity Types ===")
            for t in types["entity_types"]:
                print(f"  - {t}")
            print("\n=== Relationship Types ===")
            for t in types["relationship_types"]:
                print(f"  - {t}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        return

    # List books
    if args.list_books:
        try:
            book_files = load_book_files(args.strategy)
            print(f"\n=== Books ({len(book_files)}) ===")
            for i, path in enumerate(book_files, 1):
                print(f"  {i:2}. {path.stem}")
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        return

    # Re-consolidate only
    if args.reconsolidate:
        log_file = setup_file_logging()
        logger.info("=" * 60)
        logger.info(f"Re-Consolidating ({args.reconsolidate.upper()})")
        logger.info("=" * 60)

        try:
            results = reconsolidate(strategy=args.reconsolidate, model=args.model)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

        logger.info(f"Entity types: {len(results['consolidated_types']['entity_types'])}")
        for t in results["consolidated_types"]["entity_types"]:
            logger.info(f"  - {t}")
        logger.info(f"Saved to: {results['types_path']}")
        return

    # Full auto-tuning
    log_file = setup_file_logging()

    logger.info("=" * 60)
    logger.info("Stage 4.5: Auto-Tune Entity Types")
    logger.info("=" * 60)
    logger.info(f"Consolidation: {args.consolidation}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info(f"Model: {args.model}")

    overwrite_context = OverwriteContext(parse_overwrite_arg(args.overwrite))

    try:
        results = run_auto_tuning(
            strategy=args.strategy,
            consolidation_strategy=args.consolidation,
            overwrite_context=overwrite_context,
            model=args.model,
            skip_consolidation=args.skip_consolidation,
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted. Resume with: --overwrite skip")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

    # Summary
    logger.info("=" * 60)
    logger.info("Complete")
    logger.info("=" * 60)
    logger.info(f"Books: {len(results['processed_books'])} processed, {len(results['skipped_books'])} skipped")
    logger.info(f"Entities: {results['stats']['total_entities']:,}")

    if "consolidated_types" in results:
        logger.info(f"Entity types: {len(results['consolidated_types']['entity_types'])}")
        for t in results["consolidated_types"]["entity_types"]:
            logger.info(f"  - {t}")

    logger.info(f"Output: {results['extraction_path']}")
    if "types_path" in results:
        logger.info(f"Types: {results['types_path']}")


if __name__ == "__main__":
    main()
