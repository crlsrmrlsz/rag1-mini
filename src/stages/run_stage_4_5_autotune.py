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
# Full auto-tuning (all chunks, ~$2.86)
python -m src.stages.run_stage_4_5_autotune

# Test with limited chunks
python -m src.stages.run_stage_4_5_autotune --max-chunks 10

# Skip consolidation (just extraction)
python -m src.stages.run_stage_4_5_autotune --skip-consolidation
```

## Output

- data/processed/05_final_chunks/graph/extraction_results.json
- data/processed/05_final_chunks/graph/discovered_types.json
"""

import argparse
import sys

from src.graph.auto_tuning import run_auto_tuning, load_discovered_types
from src.shared.files import setup_logging
from src.config import GRAPHRAG_EXTRACTION_MODEL

logger = setup_logging(__name__)


def main():
    """Run auto-tuning stage."""
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
        "--max-chunks",
        type=int,
        default=None,
        help="Limit chunks for testing (default: all)",
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

    # Run auto-tuning
    logger.info("=" * 60)
    logger.info("Stage 4.5: Auto-Tune Entity Types")
    logger.info("=" * 60)

    if args.max_chunks:
        logger.info(f"Limited to {args.max_chunks} chunks (testing mode)")

    results = run_auto_tuning(
        strategy=args.strategy,
        max_chunks=args.max_chunks,
        model=args.model,
        skip_consolidation=args.skip_consolidation,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Auto-Tuning Complete")
    logger.info("=" * 60)
    logger.info(f"Chunks processed: {results['stats']['processed_chunks']}")
    logger.info(f"Entities extracted: {results['stats']['total_entities']}")
    logger.info(f"Relationships extracted: {results['stats']['total_relationships']}")
    logger.info(f"Unique entity types: {results['stats']['unique_entity_types']}")
    logger.info(f"Unique relationship types: {results['stats']['unique_relationship_types']}")

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
    logger.info(f"Extraction results: {results['extraction_path']}")
    if "types_path" in results:
        logger.info(f"Discovered types: {results['types_path']}")


if __name__ == "__main__":
    main()
