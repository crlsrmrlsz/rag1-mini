"""Auto-tuning module for GraphRAG entity type discovery.

## RAG Theory: Auto-Tuned Entity Extraction

Instead of manually defining entity types, this module discovers them
from the actual corpus content. The process:

1. Open-Ended Extraction: LLM extracts entities with freely-assigned types
2. Type Aggregation: Collect all unique types across the corpus
3. Type Consolidation: LLM proposes a clean taxonomy from discovered types
4. Save & Reuse: Store discovered types for extraction and query-time use

This approach ensures entity types match actual corpus content, improving
both extraction quality and query-time entity matching.

## Library Usage

Uses call_structured_completion() for reliable JSON parsing.
Processes chunks in batches for progress tracking.

## Data Flow

1. Load chunks from Stage 4 output
2. Extract entities with open-ended typing
3. Aggregate all entity_type values
4. LLM consolidates into taxonomy
5. Save to discovered_types.json
"""

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import Counter
import json
import time

from pydantic import BaseModel, Field

from src.config import (
    GRAPHRAG_EXTRACTION_MODEL,
    GRAPHRAG_MAX_EXTRACTION_TOKENS,
    DIR_GRAPH_DATA,
    DIR_FINAL_CHUNKS,
)
from src.shared.openrouter_client import call_structured_completion
from src.shared.files import setup_logging, OverwriteContext
from .extractor import load_chunks_for_extraction

logger = setup_logging(__name__)


# ============================================================================
# Schemas for Open-Ended Extraction
# ============================================================================

class OpenEntity(BaseModel):
    """Entity with freely-assigned type (not constrained to predefined list)."""
    name: str = Field(
        ...,
        description="Entity name as it appears in the text",
        min_length=1,
    )
    entity_type: str = Field(
        ...,
        description="Most appropriate type for this entity (e.g., BRAIN_REGION, PHILOSOPHER, CONCEPT, EMOTION)",
    )
    description: str = Field(
        default="",
        description="Brief description of this entity in context (1-2 sentences)",
    )


class OpenRelationship(BaseModel):
    """Relationship with freely-assigned type."""
    source_entity: str = Field(..., description="Name of the source entity")
    target_entity: str = Field(..., description="Name of the target entity")
    relationship_type: str = Field(
        ...,
        description="Most appropriate relationship type (e.g., CAUSES, MODULATES, PROPOSES)",
    )
    description: str = Field(default="", description="Brief description")
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class OpenExtractionResult(BaseModel):
    """Result of open-ended entity/relationship extraction."""
    entities: List[OpenEntity] = Field(default_factory=list)
    relationships: List[OpenRelationship] = Field(default_factory=list)


class ConsolidatedTypes(BaseModel):
    """LLM-consolidated entity and relationship types."""
    entity_types: List[str] = Field(
        ...,
        description="Consolidated list of entity types (15-25 types)",
    )
    relationship_types: List[str] = Field(
        ...,
        description="Consolidated list of relationship types (10-20 types)",
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of the consolidation decisions",
    )


# ============================================================================
# Open-Ended Extraction Prompt
# ============================================================================

OPEN_EXTRACTION_PROMPT = """Extract entities and relationships from this text.

For each entity, assign the MOST APPROPRIATE TYPE that describes it.
Common types include (but are not limited to):
- Brain regions: BRAIN_REGION, NEURAL_STRUCTURE
- Chemicals: NEUROTRANSMITTER, HORMONE, CHEMICAL
- Concepts: CONCEPT, THEORY, PRINCIPLE
- People: PHILOSOPHER, RESEARCHER, HISTORICAL_FIGURE
- Processes: COGNITIVE_PROCESS, BEHAVIOR, EMOTION
- Works: BOOK, STUDY, EXPERIMENT

You may create NEW types if none of the above fit well.
Use UPPERCASE_SNAKE_CASE for type names.

For relationships, use types like: CAUSES, INHIBITS, MODULATES, PROPOSES, INFLUENCES, etc.

LIMITS: Extract up to 15 most important entities and 10 relationships maximum.
Focus on the most significant concepts, not every noun.

Text:
{text}

IMPORTANT: Respond ONLY with valid JSON matching this schema:
{{"entities": [{{"name": "...", "entity_type": "...", "description": "..."}}], "relationships": [{{"source_entity": "...", "target_entity": "...", "relationship_type": "...", "description": "...", "weight": 1.0}}]}}"""


# ============================================================================
# Type Consolidation Prompt
# ============================================================================

CONSOLIDATION_PROMPT = """Analyze these entity and relationship types discovered from a corpus.

DISCOVERED ENTITY TYPES (with counts):
{entity_types}

DISCOVERED RELATIONSHIP TYPES (with counts):
{relationship_types}

Your task:
1. Consolidate similar types (e.g., BRAIN_REGION and NEURAL_STRUCTURE could merge)
2. Remove types with count=1 unless clearly important
3. Propose a clean taxonomy of 15-25 entity types and 10-20 relationship types
4. Keep types that are domain-specific and useful

IMPORTANT: Respond ONLY with valid JSON matching this schema:
{{"entity_types": ["TYPE1", "TYPE2", ...], "relationship_types": ["REL1", "REL2", ...], "rationale": "Brief explanation..."}}"""


# ============================================================================
# Main Functions
# ============================================================================

def extract_open_ended(
    chunk: Dict[str, Any],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> OpenExtractionResult:
    """Extract entities/relationships with freely-assigned types.

    Unlike standard extraction (which uses predefined types), this allows
    the LLM to assign whatever type seems most appropriate.

    Args:
        chunk: Chunk dict with 'text' field.
        model: OpenRouter model ID.

    Returns:
        OpenExtractionResult with entities and relationships.
    """
    prompt = OPEN_EXTRACTION_PROMPT.format(text=chunk["text"])
    messages = [{"role": "user", "content": prompt}]

    result = call_structured_completion(
        messages=messages,
        model=model,
        response_model=OpenExtractionResult,
        temperature=0.0,
        max_tokens=GRAPHRAG_MAX_EXTRACTION_TOKENS,
    )

    return result


def run_open_extraction(
    chunks: List[Dict[str, Any]],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Run open-ended extraction on multiple chunks.

    Args:
        chunks: List of chunk dicts.
        model: OpenRouter model ID.
        max_chunks: Optional limit for testing.

    Returns:
        Dict with entities, relationships, discovered_types, and stats.
    """
    if max_chunks:
        chunks = chunks[:max_chunks]

    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    entity_type_counter: Counter = Counter()
    relationship_type_counter: Counter = Counter()
    failed_chunks = 0

    logger.info(f"Running open-ended extraction on {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        try:
            result = extract_open_ended(chunk, model=model)

            # Add source chunk ID and collect types
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            for entity in result.entities:
                entity_dict = entity.model_dump()
                entity_dict["source_chunk_id"] = chunk_id
                all_entities.append(entity_dict)
                entity_type_counter[entity.entity_type] += 1

            for rel in result.relationships:
                rel_dict = rel.model_dump()
                rel_dict["source_chunk_id"] = chunk_id
                all_relationships.append(rel_dict)
                relationship_type_counter[rel.relationship_type] += 1

            if (i + 1) % 50 == 0:
                logger.info(
                    f"Processed {i + 1}/{len(chunks)} chunks, "
                    f"{len(all_entities)} entities, {len(entity_type_counter)} unique types"
                )

            # Rate limit protection: 1.5s delay = 40 RPM (under Tier 1 limit of 50 RPM)
            time.sleep(1.5)

        except Exception as e:
            logger.warning(f"Failed chunk {chunk.get('chunk_id', i)}: {e}")
            failed_chunks += 1
            continue

    stats = {
        "total_chunks": len(chunks),
        "processed_chunks": len(chunks) - failed_chunks,
        "failed_chunks": failed_chunks,
        "total_entities": len(all_entities),
        "total_relationships": len(all_relationships),
        "unique_entity_types": len(entity_type_counter),
        "unique_relationship_types": len(relationship_type_counter),
    }

    logger.info(f"Extraction complete: {stats}")

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": dict(entity_type_counter.most_common()),
        "relationship_type_counts": dict(relationship_type_counter.most_common()),
        "stats": stats,
    }


def consolidate_types(
    entity_type_counts: Dict[str, int],
    relationship_type_counts: Dict[str, int],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> ConsolidatedTypes:
    """Use LLM to consolidate discovered types into a clean taxonomy.

    Args:
        entity_type_counts: Dict mapping entity type to count.
        relationship_type_counts: Dict mapping relationship type to count.
        model: OpenRouter model ID.

    Returns:
        ConsolidatedTypes with clean taxonomy.
    """
    # Format types with counts for the prompt
    entity_str = "\n".join(
        f"  - {t}: {c}" for t, c in sorted(entity_type_counts.items(), key=lambda x: -x[1])
    )
    rel_str = "\n".join(
        f"  - {t}: {c}" for t, c in sorted(relationship_type_counts.items(), key=lambda x: -x[1])
    )

    prompt = CONSOLIDATION_PROMPT.format(
        entity_types=entity_str,
        relationship_types=rel_str,
    )
    messages = [{"role": "user", "content": prompt}]

    logger.info("Consolidating discovered types with LLM...")

    result = call_structured_completion(
        messages=messages,
        model=model,
        response_model=ConsolidatedTypes,
        temperature=0.0,
        max_tokens=2000,
    )

    logger.info(
        f"Consolidated to {len(result.entity_types)} entity types, "
        f"{len(result.relationship_types)} relationship types"
    )

    return result


def save_discovered_types(
    consolidated: ConsolidatedTypes,
    raw_entity_counts: Dict[str, int],
    raw_relationship_counts: Dict[str, int],
    output_name: str = "discovered_types.json",
) -> Path:
    """Save discovered types to JSON file.

    Args:
        consolidated: LLM-consolidated types.
        raw_entity_counts: Original discovered entity type counts.
        raw_relationship_counts: Original discovered relationship type counts.
        output_name: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = DIR_GRAPH_DATA
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name

    data = {
        "consolidated_entity_types": consolidated.entity_types,
        "consolidated_relationship_types": consolidated.relationship_types,
        "consolidation_rationale": consolidated.rationale,
        "raw_entity_type_counts": raw_entity_counts,
        "raw_relationship_type_counts": raw_relationship_counts,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved discovered types to {output_path}")
    return output_path


def load_discovered_types(
    file_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Load previously discovered types from JSON file.

    Args:
        file_path: Path to discovered_types.json. If None, uses default.

    Returns:
        Dict with 'entity_types' and 'relationship_types' lists.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if file_path is None:
        file_path = DIR_GRAPH_DATA / "discovered_types.json"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Discovered types not found at {file_path}. "
            "Run auto-tuning first with: python -m src.stages.run_stage_4_5_autotune"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "entity_types": data["consolidated_entity_types"],
        "relationship_types": data["consolidated_relationship_types"],
    }


def run_auto_tuning(
    strategy: str = "section",
    book_ids: Optional[List[str]] = None,
    max_chunks: Optional[int] = None,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
    skip_consolidation: bool = False,
) -> Dict[str, Any]:
    """Run full auto-tuning pipeline.

    This is the main entry point for entity type discovery:
    1. Load chunks
    2. Run open-ended extraction
    3. Consolidate types
    4. Save results

    Args:
        strategy: Chunking strategy subfolder.
        book_ids: Optional list of book IDs to process.
        max_chunks: Optional limit for testing.
        model: LLM model for extraction.
        skip_consolidation: If True, skip LLM consolidation step.

    Returns:
        Dict with all results and file paths.
    """
    # Load chunks
    chunks = load_chunks_for_extraction(strategy=strategy, book_ids=book_ids)
    logger.info(f"Loaded {len(chunks)} chunks for auto-tuning")

    # Run open-ended extraction
    results = run_open_extraction(chunks, model=model, max_chunks=max_chunks)

    # Save raw extraction results
    raw_output_path = DIR_GRAPH_DATA / "extraction_results.json"
    DIR_GRAPH_DATA.mkdir(parents=True, exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "entities": results["entities"],
            "relationships": results["relationships"],
            "stats": results["stats"],
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved extraction results to {raw_output_path}")

    # Consolidate types (optional)
    if not skip_consolidation:
        consolidated = consolidate_types(
            results["entity_type_counts"],
            results["relationship_type_counts"],
            model=model,
        )
        types_path = save_discovered_types(
            consolidated,
            results["entity_type_counts"],
            results["relationship_type_counts"],
        )
        results["consolidated_types"] = {
            "entity_types": consolidated.entity_types,
            "relationship_types": consolidated.relationship_types,
            "rationale": consolidated.rationale,
        }
        results["types_path"] = str(types_path)

    results["extraction_path"] = str(raw_output_path)
    return results


# ============================================================================
# Per-Book Resumable Functions
# ============================================================================

def load_book_files(strategy: str = "section") -> List[Path]:
    """Get list of book chunk files for per-book processing.

    Args:
        strategy: Chunking strategy subfolder (default: "section").

    Returns:
        Sorted list of book JSON file paths.

    Raises:
        FileNotFoundError: If chunk directory doesn't exist.
    """
    chunk_dir = DIR_FINAL_CHUNKS / strategy

    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    return sorted(chunk_dir.glob("*.json"))


def load_chunks_from_book(book_path: Path) -> List[Dict[str, Any]]:
    """Load chunks from a single book's JSON file.

    Args:
        book_path: Path to book's chunk JSON file.

    Returns:
        List of chunk dicts.
    """
    with open(book_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list format and dict with "chunks" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    else:
        logger.warning(f"Unexpected format in {book_path}")
        return []


def extract_book(
    book_path: Path,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> Dict[str, Any]:
    """Extract entities/relationships from a single book's chunks.

    Processes all chunks in the book with rate limiting.
    Returns results that can be saved atomically per-book.

    Args:
        book_path: Path to book's chunk JSON file.
        model: OpenRouter model ID.

    Returns:
        Dict with entities, relationships, type counts, and stats.
    """
    book_name = book_path.stem
    chunks = load_chunks_from_book(book_path)

    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    entity_type_counter: Counter = Counter()
    relationship_type_counter: Counter = Counter()
    failed_chunks = 0

    logger.info(f"Extracting from {book_name}: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        try:
            result = extract_open_ended(chunk, model=model)

            # Add source chunk ID and collect types
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            for entity in result.entities:
                entity_dict = entity.model_dump()
                entity_dict["source_chunk_id"] = chunk_id
                all_entities.append(entity_dict)
                entity_type_counter[entity.entity_type] += 1

            for rel in result.relationships:
                rel_dict = rel.model_dump()
                rel_dict["source_chunk_id"] = chunk_id
                all_relationships.append(rel_dict)
                relationship_type_counter[rel.relationship_type] += 1

            if (i + 1) % 20 == 0:
                logger.info(
                    f"  [{book_name}] {i + 1}/{len(chunks)} chunks, "
                    f"{len(all_entities)} entities"
                )

            # Rate limit protection: 1.5s delay = 40 RPM (under Tier 1 limit of 50 RPM)
            time.sleep(1.5)

        except Exception as e:
            logger.warning(f"Failed chunk {chunk.get('chunk_id', i)}: {e}")
            failed_chunks += 1
            continue

    stats = {
        "book_name": book_name,
        "total_chunks": len(chunks),
        "processed_chunks": len(chunks) - failed_chunks,
        "failed_chunks": failed_chunks,
        "total_entities": len(all_entities),
        "total_relationships": len(all_relationships),
    }

    logger.info(
        f"  [{book_name}] Complete: {stats['processed_chunks']}/{stats['total_chunks']} chunks, "
        f"{len(all_entities)} entities, {len(all_relationships)} relationships"
    )

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": dict(entity_type_counter.most_common()),
        "relationship_type_counts": dict(relationship_type_counter.most_common()),
        "stats": stats,
    }


def merge_book_extractions(extractions_dir: Path) -> Dict[str, Any]:
    """Merge all per-book extraction files into aggregated results.

    Reads all {book}.json files from extractions_dir and combines them.

    Args:
        extractions_dir: Directory containing per-book extraction JSON files.

    Returns:
        Dict with aggregated entities, relationships, type counts, and stats.
    """
    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    entity_type_counter: Counter = Counter()
    relationship_type_counter: Counter = Counter()
    total_stats = {
        "total_books": 0,
        "total_chunks": 0,
        "processed_chunks": 0,
        "failed_chunks": 0,
        "total_entities": 0,
        "total_relationships": 0,
    }

    extraction_files = sorted(extractions_dir.glob("*.json"))
    logger.info(f"Merging {len(extraction_files)} book extraction files...")

    for extraction_file in extraction_files:
        with open(extraction_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_entities.extend(data.get("entities", []))
        all_relationships.extend(data.get("relationships", []))

        # Aggregate type counts
        for etype, count in data.get("entity_type_counts", {}).items():
            entity_type_counter[etype] += count
        for rtype, count in data.get("relationship_type_counts", {}).items():
            relationship_type_counter[rtype] += count

        # Aggregate stats
        book_stats = data.get("stats", {})
        total_stats["total_books"] += 1
        total_stats["total_chunks"] += book_stats.get("total_chunks", 0)
        total_stats["processed_chunks"] += book_stats.get("processed_chunks", 0)
        total_stats["failed_chunks"] += book_stats.get("failed_chunks", 0)

    total_stats["total_entities"] = len(all_entities)
    total_stats["total_relationships"] = len(all_relationships)
    total_stats["unique_entity_types"] = len(entity_type_counter)
    total_stats["unique_relationship_types"] = len(relationship_type_counter)

    logger.info(f"Merged results: {total_stats}")

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": dict(entity_type_counter.most_common()),
        "relationship_type_counts": dict(relationship_type_counter.most_common()),
        "stats": total_stats,
    }


def run_auto_tuning_resumable(
    strategy: str = "section",
    overwrite_context: Optional[OverwriteContext] = None,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
    skip_consolidation: bool = False,
) -> Dict[str, Any]:
    """Run auto-tuning with per-book resume support.

    Processes each book atomically, saving results per-book.
    Uses OverwriteContext to check if book already processed.
    After all books, merges into final extraction_results.json.

    Args:
        strategy: Chunking strategy subfolder.
        overwrite_context: Context for overwrite decisions. If None, overwrites all.
        model: LLM model for extraction.
        skip_consolidation: If True, skip LLM consolidation step.

    Returns:
        Dict with all results and file paths.
    """
    book_files = load_book_files(strategy)
    logger.info(f"Found {len(book_files)} books to process")

    # Setup output directory for per-book extractions
    extractions_dir = DIR_GRAPH_DATA / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    processed_books: List[str] = []
    skipped_books: List[str] = []

    for book_path in book_files:
        book_name = book_path.stem
        output_path = extractions_dir / f"{book_name}.json"

        # Check overwrite decision
        if overwrite_context and not overwrite_context.should_overwrite(output_path, logger):
            skipped_books.append(book_name)
            continue

        # Extract from this book
        try:
            results = extract_book(book_path, model=model)

            # Save atomically
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            processed_books.append(book_name)
            logger.info(f"Saved: {output_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {book_name}: {e}")
            raise  # Fail-fast: stop on first error

    logger.info(
        f"Book processing complete: {len(processed_books)} processed, "
        f"{len(skipped_books)} skipped"
    )

    # Merge all book extractions
    merged_results = merge_book_extractions(extractions_dir)

    # Save merged extraction results
    raw_output_path = DIR_GRAPH_DATA / "extraction_results.json"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "entities": merged_results["entities"],
            "relationships": merged_results["relationships"],
            "stats": merged_results["stats"],
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved merged extraction results to {raw_output_path}")

    # Consolidate types (optional)
    if not skip_consolidation:
        consolidated = consolidate_types(
            merged_results["entity_type_counts"],
            merged_results["relationship_type_counts"],
            model=model,
        )
        types_path = save_discovered_types(
            consolidated,
            merged_results["entity_type_counts"],
            merged_results["relationship_type_counts"],
        )
        merged_results["consolidated_types"] = {
            "entity_types": consolidated.entity_types,
            "relationship_types": consolidated.relationship_types,
            "rationale": consolidated.rationale,
        }
        merged_results["types_path"] = str(types_path)

    merged_results["extraction_path"] = str(raw_output_path)
    merged_results["extractions_dir"] = str(extractions_dir)
    merged_results["processed_books"] = processed_books
    merged_results["skipped_books"] = skipped_books

    return merged_results
