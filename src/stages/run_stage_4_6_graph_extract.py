"""Stage 4.6: Entity and Relationship Extraction for GraphRAG.

## RAG Theory: Entity Extraction

This stage extracts entities and relationships from text chunks using LLM.
The extracted graph serves as the foundation for:
- Knowledge graph construction (Neo4j)
- Leiden community detection
- Hybrid retrieval (graph + vector)

## Data Flow

Input: Section chunks from Stage 4 (data/processed/05_final_chunks/{strategy}/)
Output: Extraction results (data/processed/07_graph/extraction_results.json)

## Usage

```bash
# Extract from all section chunks
python -m src.stages.run_stage_4_6_graph_extract

# Limit to specific number of chunks (for testing)
python -m src.stages.run_stage_4_6_graph_extract --max-chunks 10

# Use specific strategy
python -m src.stages.run_stage_4_6_graph_extract --strategy contextual
```
"""

import argparse
import time
from pathlib import Path

from src.shared.files import setup_logging
from src.graph.extractor import run_extraction
from src.config import GRAPHRAG_EXTRACTION_MODEL

logger = setup_logging(__name__)


def main():
    """Run entity extraction stage."""
    parser = argparse.ArgumentParser(
        description="Stage 4.6: Extract entities and relationships for GraphRAG"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="section",
        help="Chunking strategy to use (default: section)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process (for testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GRAPHRAG_EXTRACTION_MODEL,
        help=f"LLM model for extraction (default: {GRAPHRAG_EXTRACTION_MODEL})",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STAGE 4.6: GRAPHRAG ENTITY EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Model: {args.model}")
    if args.max_chunks:
        logger.info(f"Max chunks: {args.max_chunks}")

    start_time = time.time()

    # Run extraction
    try:
        results = run_extraction(
            strategy=args.strategy,
            max_chunks=args.max_chunks,
            model=args.model,
        )

        elapsed = time.time() - start_time

        # Print summary
        logger.info("-" * 60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("-" * 60)
        logger.info(f"Chunks processed: {results['stats']['processed_chunks']}")
        logger.info(f"Entities extracted: {results['stats']['total_entities']}")
        logger.info(f"Relationships extracted: {results['stats']['total_relationships']}")
        logger.info(f"Unique entity types: {results['stats']['unique_entity_types']}")
        logger.info(f"Unique relationship types: {results['stats']['unique_relationship_types']}")
        logger.info(f"Failed chunks: {results['stats']['failed_chunks']}")
        logger.info(f"Time elapsed: {elapsed:.1f}s")
        logger.info(f"Output: {results['output_path']}")

    except FileNotFoundError as e:
        logger.error(f"Chunk files not found: {e}")
        logger.error("Run Stage 4 (chunking) first: python -m src.stages.run_stage_4_chunking")
        raise

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
