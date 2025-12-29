"""Stage 6b: Neo4j Upload and Leiden Community Detection.

## RAG Theory: Knowledge Graph Construction

This stage uploads extracted entities and relationships to Neo4j,
then runs Leiden community detection to identify clusters of
related entities. Community summaries enable global queries.

## Crash-Proof Design

Stage 6b is designed to handle crashes during the ~10 hour summarization:

1. **Deterministic Leiden**: Uses randomSeed=42 + concurrency=1
   - Same graph + same seed = same community assignments (guaranteed)
   - Enables resume after Neo4j reset without ID mismatches

2. **Weaviate Storage**: Community embeddings stored in Weaviate
   - Efficient HNSW vector search (O(log n) vs O(n) for JSON file)
   - ~12MB total vs 383MB JSON with inline embeddings

3. **Atomic Uploads**: Each community uploaded to Weaviate immediately
   - Resume skips existing communities (checks Weaviate)
   - No data loss on crash

## Data Flow

Input: Extraction results from Stage 4.6 (data/processed/07_graph/extraction_results.json)
Output:
- Neo4j graph with entities, relationships, and community IDs
- Weaviate collection with community embeddings (Community_section800_v1)
- Leiden checkpoint (data/processed/07_graph/leiden_checkpoint.json)
- Backup JSON (data/processed/07_graph/communities.json)

## Prerequisites

1. Neo4j must be running: docker compose up -d neo4j
2. Weaviate must be running: docker compose up -d weaviate
3. Stage 4.6 must be complete (extraction results exist)

## Usage

```bash
# Full pipeline: upload + Leiden + summarization
python -m src.stages.run_stage_6b_neo4j

# Upload only (skip Leiden)
python -m src.stages.run_stage_6b_neo4j --upload-only

# Leiden only (assumes graph exists)
python -m src.stages.run_stage_6b_neo4j --leiden-only

# Clear graph before upload
python -m src.stages.run_stage_6b_neo4j --clear

# Resume after crash (skip Leiden, continue from Weaviate checkpoint)
python -m src.stages.run_stage_6b_neo4j --resume
```
"""

import argparse
import json
import time
from pathlib import Path

from src.shared.files import setup_logging
from src.config import DIR_GRAPH_DATA, GRAPHRAG_SUMMARY_MODEL
from src.graph.neo4j_client import (
    get_driver,
    get_gds_client,
    verify_connection,
    clear_graph,
    upload_extraction_results,
    get_graph_stats,
)
from src.graph.community import (
    detect_and_summarize_communities,
    save_communities,
    get_community_ids_from_neo4j,
)

logger = setup_logging(__name__)


def load_extraction_results(
    input_name: str = "extraction_results.json",
) -> dict:
    """Load extraction results from JSON file.

    Args:
        input_name: Input filename.

    Returns:
        Dict with entities and relationships.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    input_path = DIR_GRAPH_DATA / input_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Extraction results not found: {input_path}\n"
            "Run Stage 4.6 first: python -m src.stages.run_stage_4_6_graph_extract"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Run Neo4j upload and Leiden community detection."""
    parser = argparse.ArgumentParser(
        description="Stage 6b: Upload to Neo4j and run Leiden community detection"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload entities/relationships, skip Leiden",
    )
    parser.add_argument(
        "--leiden-only",
        action="store_true",
        help="Only run Leiden (assumes graph exists)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing graph before upload",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GRAPHRAG_SUMMARY_MODEL,
        help=f"LLM model for community summarization (default: {GRAPHRAG_SUMMARY_MODEL})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume after crash: skip Leiden, generate missing summaries (checks Weaviate)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STAGE 6b: NEO4J UPLOAD + LEIDEN COMMUNITIES")
    logger.info("=" * 60)

    start_time = time.time()

    # Connect to Neo4j
    try:
        driver = get_driver()
        verify_connection(driver)
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.error("Ensure Neo4j is running: docker compose up -d neo4j")
        raise

    try:
        # Validate prerequisites for resume mode
        if args.resume:
            logger.info("-" * 60)
            logger.info("RESUME MODE: Validating prerequisites")
            logger.info("-" * 60)

            # Check entities exist
            stats = get_graph_stats(driver)
            if stats["node_count"] == 0:
                raise ValueError(
                    "Cannot resume: no entities in Neo4j. "
                    "Run full pipeline first: python -m src.stages.run_stage_6b_neo4j"
                )

            # Check community_ids exist (Leiden must have completed)
            community_ids = get_community_ids_from_neo4j(driver)
            if not community_ids:
                raise ValueError(
                    "Cannot resume: no community_ids found in Neo4j. "
                    "Leiden may not have completed. Run full pipeline first."
                )

            logger.info(f"Validation passed: {stats['node_count']} entities, {len(community_ids)} communities")

        # Phase 1: Upload to Neo4j (unless leiden-only or resume)
        if not args.leiden_only and not args.resume:
            logger.info("-" * 60)
            logger.info("PHASE 1: UPLOAD TO NEO4J")
            logger.info("-" * 60)

            # Load extraction results
            results = load_extraction_results()
            logger.info(
                f"Loaded {len(results['entities'])} entities, "
                f"{len(results['relationships'])} relationships"
            )

            # Clear graph if requested
            if args.clear:
                clear_graph(driver)

            # Upload
            upload_start = time.time()
            counts = upload_extraction_results(driver, results)
            upload_time = time.time() - upload_start

            logger.info(f"Upload complete in {upload_time:.1f}s")
            logger.info(f"  Entities: {counts['entity_count']}")
            logger.info(f"  Relationships: {counts['relationship_count']}")

        # Get graph stats
        stats = get_graph_stats(driver)
        logger.info(f"Graph stats: {stats['node_count']} nodes, {stats['relationship_count']} relationships")

        # Phase 2: Leiden community detection (unless upload-only)
        if not args.upload_only:
            logger.info("-" * 60)
            logger.info("PHASE 2: LEIDEN COMMUNITY DETECTION")
            logger.info("-" * 60)

            if stats["node_count"] == 0:
                logger.warning("Graph is empty, skipping Leiden")
            else:
                # Get GDS client
                gds = get_gds_client(driver)

                # Run Leiden and generate summaries
                leiden_start = time.time()
                communities = detect_and_summarize_communities(
                    driver,
                    gds,
                    model=args.model,
                    resume=args.resume,
                    skip_leiden=args.resume,  # Skip Leiden if resuming (community_ids exist)
                )
                leiden_time = time.time() - leiden_start

                # Save communities
                output_path = save_communities(communities)

                logger.info(f"Leiden complete in {leiden_time:.1f}s")
                logger.info(f"  Communities found: {len(communities)}")
                logger.info(f"  Total members: {sum(c.member_count for c in communities)}")
                logger.info(f"  Output: {output_path}")

        # Final summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("STAGE 6b COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.1f}s")

        # Print final graph stats
        final_stats = get_graph_stats(driver)
        logger.info(f"Final graph: {final_stats['node_count']} nodes, {final_stats['relationship_count']} relationships")

        if final_stats.get("entity_types"):
            logger.info("Entity types:")
            for etype, count in list(final_stats["entity_types"].items())[:5]:
                logger.info(f"  {etype}: {count}")

    finally:
        driver.close()
        logger.info("Neo4j connection closed")


if __name__ == "__main__":
    main()
