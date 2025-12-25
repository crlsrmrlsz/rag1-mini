"""LLM-based entity and relationship extraction for GraphRAG.

## RAG Theory: Entity Extraction in GraphRAG

Entity extraction is the first stage of GraphRAG's indexing pipeline.
For each text chunk, an LLM identifies:
1. Named entities (people, concepts, brain regions, etc.)
2. Relationships between entities (causes, influences, etc.)

The extraction uses structured output (JSON Schema) to ensure
reliable parsing. Domain-specific entity types from config guide
the LLM to extract relevant entities for the corpus.

## Library Usage

Uses call_structured_completion() from openrouter_client.py:
- Pydantic model → JSON Schema → OpenRouter response_format
- Automatic validation and parsing of LLM output
- Retry logic with exponential backoff

## Data Flow

1. Load section chunks from Stage 4 output
2. For each chunk, call LLM with extraction prompt
3. Parse response into ExtractionResult (entities + relationships)
4. Aggregate results for Neo4j upload
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from src.config import (
    GRAPHRAG_ENTITY_TYPES,
    GRAPHRAG_RELATIONSHIP_TYPES,
    GRAPHRAG_EXTRACTION_MODEL,
    GRAPHRAG_EXTRACTION_PROMPT,
    GRAPHRAG_MAX_EXTRACTION_TOKENS,
    DIR_FINAL_CHUNKS,
    DIR_GRAPH_DATA,
)
from src.shared.openrouter_client import call_structured_completion
from src.shared.files import setup_logging
from .schemas import GraphEntity, GraphRelationship, ExtractionResult

logger = setup_logging(__name__)


def extract_from_chunk(
    chunk: Dict[str, Any],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> ExtractionResult:
    """Extract entities and relationships from a single chunk.

    Uses structured output to ensure valid JSON response from LLM.
    Entity types and relationship types from config guide extraction.

    Args:
        chunk: Chunk dict with 'text', 'chunk_id', 'context' fields.
        model: OpenRouter model ID for extraction.

    Returns:
        ExtractionResult with entities and relationships.

    Raises:
        OpenRouterError: If LLM call fails after retries.
        ValidationError: If response doesn't match schema (rare with structured output).

    Example:
        >>> chunk = {"text": "The prefrontal cortex regulates emotion...", "chunk_id": "x"}
        >>> result = extract_from_chunk(chunk)
        >>> len(result.entities)
        2
    """
    # Build the extraction prompt
    prompt = GRAPHRAG_EXTRACTION_PROMPT.format(
        entity_types=", ".join(GRAPHRAG_ENTITY_TYPES),
        relationship_types=", ".join(GRAPHRAG_RELATIONSHIP_TYPES),
        text=chunk["text"],
    )

    messages = [{"role": "user", "content": prompt}]

    # Call LLM with structured output
    result = call_structured_completion(
        messages=messages,
        model=model,
        response_model=ExtractionResult,
        temperature=0.0,  # Deterministic for reproducibility
        max_tokens=GRAPHRAG_MAX_EXTRACTION_TOKENS,
    )

    # Add source chunk ID to all entities and relationships
    chunk_id = chunk.get("chunk_id", "")
    for entity in result.entities:
        entity.source_chunk_id = chunk_id
    for rel in result.relationships:
        rel.source_chunk_id = chunk_id

    return result


def extract_from_chunks(
    chunks: List[Dict[str, Any]],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract entities and relationships from multiple chunks.

    Processes chunks sequentially (LLM calls are the bottleneck).
    Aggregates all entities and relationships, tracking source chunks.

    Args:
        chunks: List of chunk dicts from Stage 4.
        model: OpenRouter model ID for extraction.
        max_chunks: Optional limit for testing (None = all chunks).

    Returns:
        Dict with:
        - entities: List of GraphEntity dicts
        - relationships: List of GraphRelationship dicts
        - stats: Extraction statistics

    Example:
        >>> result = extract_from_chunks(chunks[:10])
        >>> print(result["stats"])
        {"total_chunks": 10, "total_entities": 45, "total_relationships": 32}
    """
    if max_chunks:
        chunks = chunks[:max_chunks]

    all_entities: List[GraphEntity] = []
    all_relationships: List[GraphRelationship] = []
    failed_chunks = 0

    logger.info(f"Extracting entities from {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        try:
            result = extract_from_chunk(chunk, model=model)
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(chunks)} chunks, "
                    f"{len(all_entities)} entities, {len(all_relationships)} relationships"
                )

        except Exception as e:
            logger.warning(f"Failed to extract from chunk {chunk.get('chunk_id', i)}: {e}")
            failed_chunks += 1
            continue

    stats = {
        "total_chunks": len(chunks),
        "processed_chunks": len(chunks) - failed_chunks,
        "failed_chunks": failed_chunks,
        "total_entities": len(all_entities),
        "total_relationships": len(all_relationships),
        "unique_entity_types": len(set(e.entity_type for e in all_entities)),
        "unique_relationship_types": len(set(r.relationship_type for r in all_relationships)),
    }

    logger.info(f"Extraction complete: {stats}")

    return {
        "entities": [e.model_dump() for e in all_entities],
        "relationships": [r.model_dump() for r in all_relationships],
        "stats": stats,
    }


def load_chunks_for_extraction(
    strategy: str = "section",
    book_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load chunks from Stage 4 for entity extraction.

    Reads chunk files from DIR_FINAL_CHUNKS/{strategy}/ directory.
    Optionally filters by book IDs.

    Args:
        strategy: Chunking strategy subfolder (default: "section").
        book_ids: Optional list of book IDs to include (None = all).

    Returns:
        List of chunk dicts with text, chunk_id, context, etc.

    Raises:
        FileNotFoundError: If chunk directory doesn't exist.
    """
    chunk_dir = DIR_FINAL_CHUNKS / strategy

    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    all_chunks = []
    chunk_files = list(chunk_dir.glob("*.json"))

    for chunk_file in chunk_files:
        # Filter by book ID if specified
        if book_ids:
            book_id = chunk_file.stem  # filename without extension
            if book_id not in book_ids:
                continue

        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both list format and dict with "chunks" key
            if isinstance(data, list):
                all_chunks.extend(data)
            elif isinstance(data, dict) and "chunks" in data:
                all_chunks.extend(data["chunks"])
            else:
                logger.warning(f"Unexpected format in {chunk_file}")

    logger.info(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files")
    return all_chunks


def save_extraction_results(
    results: Dict[str, Any],
    output_name: str = "extraction_results.json",
) -> Path:
    """Save extraction results to JSON file.

    Saves to DIR_GRAPH_DATA for later Neo4j upload.

    Args:
        results: Dict from extract_from_chunks().
        output_name: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = DIR_GRAPH_DATA
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved extraction results to {output_path}")
    return output_path


def run_extraction(
    strategy: str = "section",
    book_ids: Optional[List[str]] = None,
    max_chunks: Optional[int] = None,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> Dict[str, Any]:
    """Run full entity extraction pipeline.

    Main entry point for Stage 4.6 (graph extraction).

    Args:
        strategy: Chunking strategy to use.
        book_ids: Optional list of book IDs to process.
        max_chunks: Optional limit for testing.
        model: LLM model for extraction.

    Returns:
        Dict with entities, relationships, stats, and output_path.

    Example:
        >>> result = run_extraction(strategy="section", max_chunks=10)
        >>> print(result["stats"]["total_entities"])
        45
    """
    # Load chunks
    chunks = load_chunks_for_extraction(strategy=strategy, book_ids=book_ids)

    # Extract entities and relationships
    results = extract_from_chunks(chunks, model=model, max_chunks=max_chunks)

    # Save results
    output_path = save_extraction_results(results)
    results["output_path"] = str(output_path)

    return results
