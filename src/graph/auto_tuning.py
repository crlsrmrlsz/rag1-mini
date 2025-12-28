"""Auto-tuning module for GraphRAG entity type discovery.

## What This Module Does

1. Open-Ended Extraction: LLM extracts entities with freely-assigned types
2. Type Aggregation: Collect all unique types across the corpus
3. Type Consolidation: LLM proposes a clean taxonomy from discovered types
4. Save & Reuse: Store discovered types for query-time entity matching

## Consolidation Strategies

- **stratified** (default): Balances entity types across different corpora
  (e.g., neuroscience vs philosophy) to prevent larger corpora from dominating.
- **global**: Original algorithm, ranks by total count across all books.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter
import json
import time

from pydantic import BaseModel, Field

from src.config import (
    GRAPHRAG_EXTRACTION_MODEL,
    GRAPHRAG_MAX_EXTRACTION_TOKENS,
    GRAPHRAG_MAX_ENTITIES,
    GRAPHRAG_MAX_RELATIONSHIPS,
    DIR_GRAPH_DATA,
    DIR_FINAL_CHUNKS,
    CORPUS_BOOK_MAPPING,
    GRAPHRAG_TYPES_PER_CORPUS,
    GRAPHRAG_MIN_CORPUS_PERCENTAGE,
    GRAPHRAG_OPEN_EXTRACTION_PROMPT,
    GRAPHRAG_GLOBAL_CONSOLIDATION_PROMPT,
    GRAPHRAG_STRATIFIED_CONSOLIDATION_PROMPT,
)
from src.shared.openrouter_client import call_structured_completion
from src.shared.files import setup_logging, OverwriteContext

logger = setup_logging(__name__)


# ============================================================================
# Pydantic Schemas
# ============================================================================

class OpenEntity(BaseModel):
    """Entity with freely-assigned type."""
    name: str = Field(..., min_length=1)
    entity_type: str = Field(...)
    description: str = Field(default="")


class OpenRelationship(BaseModel):
    """Relationship with freely-assigned type."""
    source_entity: str = Field(...)
    target_entity: str = Field(...)
    relationship_type: str = Field(...)
    description: str = Field(default="")
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class OpenExtractionResult(BaseModel):
    """Result of open-ended entity/relationship extraction."""
    entities: List[OpenEntity] = Field(default_factory=list)
    relationships: List[OpenRelationship] = Field(default_factory=list)


class ConsolidatedTypes(BaseModel):
    """LLM-consolidated entity and relationship types."""
    entity_types: List[str] = Field(...)
    relationship_types: List[str] = Field(...)
    rationale: str = Field(default="")


# ============================================================================
# Core Extraction Functions
# ============================================================================

def extract_chunk(
    chunk: Dict[str, Any],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> OpenExtractionResult:
    """Extract entities/relationships from a single chunk."""
    prompt = GRAPHRAG_OPEN_EXTRACTION_PROMPT.format(
        text=chunk["text"],
        max_entities=GRAPHRAG_MAX_ENTITIES,
        max_relationships=GRAPHRAG_MAX_RELATIONSHIPS,
    )
    return call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=OpenExtractionResult,
        temperature=0.0,
        max_tokens=GRAPHRAG_MAX_EXTRACTION_TOKENS,
    )


def extract_book(
    book_path: Path,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> Dict[str, Any]:
    """Extract entities/relationships from all chunks in a book.

    Args:
        book_path: Path to book's chunk JSON file.
        model: LLM model for extraction.

    Returns:
        Dict with entities, relationships, type counts, and stats.
    """
    book_name = book_path.stem

    # Load chunks
    with open(book_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data if isinstance(data, list) else data.get("chunks", [])

    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    entity_type_counter: Counter = Counter()
    relationship_type_counter: Counter = Counter()
    failed_chunks = 0

    logger.info(f"Extracting from {book_name}: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        try:
            result = extract_chunk(chunk, model=model)
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
                logger.info(f"  [{book_name}] {i + 1}/{len(chunks)} chunks")

            time.sleep(1.5)  # Rate limit: 40 RPM

        except Exception as e:
            logger.warning(f"Failed chunk {chunk.get('chunk_id', i)}: {e}")
            failed_chunks += 1

    logger.info(
        f"  [{book_name}] Done: {len(chunks) - failed_chunks}/{len(chunks)} chunks, "
        f"{len(all_entities)} entities"
    )

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": dict(entity_type_counter.most_common()),
        "relationship_type_counts": dict(relationship_type_counter.most_common()),
        "stats": {
            "book_name": book_name,
            "total_chunks": len(chunks),
            "processed_chunks": len(chunks) - failed_chunks,
            "failed_chunks": failed_chunks,
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
        },
    }


# ============================================================================
# Consolidation Functions
# ============================================================================

def consolidate_global(
    entity_type_counts: Dict[str, int],
    relationship_type_counts: Dict[str, int],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> ConsolidatedTypes:
    """Consolidate types using global frequency ranking."""
    entity_str = "\n".join(
        f"  - {t}: {c}" for t, c in sorted(entity_type_counts.items(), key=lambda x: -x[1])
    )
    rel_str = "\n".join(
        f"  - {t}: {c}" for t, c in sorted(relationship_type_counts.items(), key=lambda x: -x[1])
    )

    prompt = GRAPHRAG_GLOBAL_CONSOLIDATION_PROMPT.format(
        entity_types=entity_str,
        relationship_types=rel_str,
    )

    logger.info("Consolidating types (global strategy)...")
    return call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=ConsolidatedTypes,
        temperature=0.0,
        max_tokens=2000,
    )


def consolidate_stratified(
    extractions_dir: Path,
    relationship_type_counts: Dict[str, int],
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> ConsolidatedTypes:
    """Consolidate types with balanced representation from each corpus.

    Prevents larger corpora from dominating by selecting top-K types
    from EACH corpus proportionally.
    """
    # Build book -> corpus mapping
    book_to_corpus = {
        book: corpus
        for corpus, books in CORPUS_BOOK_MAPPING.items()
        for book in books
    }

    # Aggregate entity types per corpus
    corpus_counts: Dict[str, Counter] = {c: Counter() for c in CORPUS_BOOK_MAPPING}
    corpus_totals: Dict[str, int] = {c: 0 for c in CORPUS_BOOK_MAPPING}

    for extraction_file in extractions_dir.glob("*.json"):
        corpus = book_to_corpus.get(extraction_file.stem)
        if not corpus:
            logger.warning(f"Book not in corpus mapping: {extraction_file.stem}")
            continue

        with open(extraction_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for etype, count in data.get("entity_type_counts", {}).items():
            corpus_counts[corpus][etype] += count
            corpus_totals[corpus] += count

    # Select top-K types per corpus
    corpus_top: Dict[str, List[Tuple[str, float]]] = {}
    for corpus, counter in corpus_counts.items():
        total = corpus_totals[corpus]
        if total == 0:
            continue
        type_pcts = [
            (etype, 100 * count / total)
            for etype, count in counter.most_common()
            if 100 * count / total >= GRAPHRAG_MIN_CORPUS_PERCENTAGE
        ]
        corpus_top[corpus] = type_pcts[:GRAPHRAG_TYPES_PER_CORPUS]
        logger.info(f"Corpus '{corpus}': {total:,} entities, top {len(corpus_top[corpus])} types")

    # Identify shared types
    all_types = {t for types in corpus_top.values() for t, _ in types}
    shared = {t for t in all_types if sum(1 for types in corpus_top.values() if any(x[0] == t for x in types)) > 1}

    # Format for prompt
    corpus_names = list(corpus_top.keys())

    def format_types(corpus: str) -> str:
        return "\n".join(
            f"  - {t}: {p:.1f}% ({corpus_counts[corpus][t]:,})"
            for t, p in corpus_top.get(corpus, []) if t not in shared
        ) or "  (none)"

    def format_shared() -> str:
        lines = []
        for t in sorted(shared):
            parts = [f"{c}: {100 * corpus_counts[c][t] / corpus_totals[c]:.1f}%"
                     for c in corpus_names if corpus_counts[c][t] > 0]
            lines.append(f"  - {t}: {', '.join(parts)}")
        return "\n".join(lines) or "  (none)"

    rel_str = "\n".join(
        f"  - {t}: {c}" for t, c in sorted(relationship_type_counts.items(), key=lambda x: -x[1])[:25]
    )

    prompt = GRAPHRAG_STRATIFIED_CONSOLIDATION_PROMPT.format(
        corpus1_name=corpus_names[0].upper() if corpus_names else "CORPUS1",
        corpus1_types=format_types(corpus_names[0]) if corpus_names else "",
        corpus2_name=corpus_names[1].upper() if len(corpus_names) > 1 else "CORPUS2",
        corpus2_types=format_types(corpus_names[1]) if len(corpus_names) > 1 else "",
        shared_types=format_shared(),
        relationship_types=rel_str,
    )

    logger.info(f"Consolidating types (stratified strategy), shared: {sorted(shared)}")
    return call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=ConsolidatedTypes,
        temperature=0.0,
        max_tokens=2000,
    )


# ============================================================================
# File I/O Functions
# ============================================================================

def load_book_files(strategy: str = "section") -> List[Path]:
    """Get list of book chunk files."""
    chunk_dir = DIR_FINAL_CHUNKS / strategy
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
    return sorted(chunk_dir.glob("*.json"))


def merge_extractions(extractions_dir: Path) -> Dict[str, Any]:
    """Merge all per-book extraction files into aggregated results."""
    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    entity_counter: Counter = Counter()
    relationship_counter: Counter = Counter()
    stats = {"total_books": 0, "processed_chunks": 0, "failed_chunks": 0}

    for f in sorted(extractions_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        all_entities.extend(data.get("entities", []))
        all_relationships.extend(data.get("relationships", []))
        for t, c in data.get("entity_type_counts", {}).items():
            entity_counter[t] += c
        for t, c in data.get("relationship_type_counts", {}).items():
            relationship_counter[t] += c
        book_stats = data.get("stats", {})
        stats["total_books"] += 1
        stats["processed_chunks"] += book_stats.get("processed_chunks", 0)
        stats["failed_chunks"] += book_stats.get("failed_chunks", 0)

    stats["total_entities"] = len(all_entities)
    stats["total_relationships"] = len(all_relationships)
    stats["unique_entity_types"] = len(entity_counter)
    stats["unique_relationship_types"] = len(relationship_counter)

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "entity_type_counts": dict(entity_counter.most_common()),
        "relationship_type_counts": dict(relationship_counter.most_common()),
        "stats": stats,
    }


def save_discovered_types(
    consolidated: ConsolidatedTypes,
    raw_entity_counts: Dict[str, int],
    raw_relationship_counts: Dict[str, int],
    consolidation_method: str = "stratified",
) -> Path:
    """Save discovered types to JSON file."""
    DIR_GRAPH_DATA.mkdir(parents=True, exist_ok=True)
    output_path = DIR_GRAPH_DATA / "discovered_types.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "consolidated_entity_types": consolidated.entity_types,
            "consolidated_relationship_types": consolidated.relationship_types,
            "consolidation_rationale": consolidated.rationale,
            "consolidation_method": consolidation_method,
            "raw_entity_type_counts": raw_entity_counts,
            "raw_relationship_type_counts": raw_relationship_counts,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved discovered types to {output_path}")
    return output_path


def load_discovered_types(file_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load previously discovered types from JSON file."""
    if file_path is None:
        file_path = DIR_GRAPH_DATA / "discovered_types.json"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Discovered types not found: {file_path}. "
            "Run: python -m src.stages.run_stage_4_5_autotune"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "entity_types": data["consolidated_entity_types"],
        "relationship_types": data["consolidated_relationship_types"],
    }


# ============================================================================
# Main Entry Points
# ============================================================================

def run_auto_tuning(
    strategy: str = "section",
    consolidation_strategy: str = "stratified",
    overwrite_context: Optional[OverwriteContext] = None,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
    skip_consolidation: bool = False,
) -> Dict[str, Any]:
    """Run auto-tuning with per-book extraction and configurable consolidation.

    Args:
        strategy: Chunking strategy subfolder (default: "section").
        consolidation_strategy: "stratified" (default) or "global".
        overwrite_context: Controls whether to skip already-processed books.
        model: LLM model for extraction.
        skip_consolidation: If True, skip LLM consolidation step.

    Returns:
        Dict with extraction results, consolidated types, and file paths.
    """
    book_files = load_book_files(strategy)
    logger.info(f"Found {len(book_files)} books")

    extractions_dir = DIR_GRAPH_DATA / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    processed, skipped = [], []

    for book_path in book_files:
        book_name = book_path.stem
        output_path = extractions_dir / f"{book_name}.json"

        if overwrite_context and not overwrite_context.should_overwrite(output_path, logger):
            skipped.append(book_name)
            continue

        results = extract_book(book_path, model=model)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        processed.append(book_name)
        logger.info(f"Saved: {output_path.name}")

    logger.info(f"Extraction complete: {len(processed)} processed, {len(skipped)} skipped")

    # Merge all extractions
    merged = merge_extractions(extractions_dir)

    # Save merged results
    extraction_path = DIR_GRAPH_DATA / "extraction_results.json"
    with open(extraction_path, "w", encoding="utf-8") as f:
        json.dump({
            "entities": merged["entities"],
            "relationships": merged["relationships"],
            "stats": merged["stats"],
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved merged results to {extraction_path}")

    # Consolidate types
    if not skip_consolidation:
        if consolidation_strategy == "stratified":
            consolidated = consolidate_stratified(
                extractions_dir,
                merged["relationship_type_counts"],
                model=model,
            )
        else:
            consolidated = consolidate_global(
                merged["entity_type_counts"],
                merged["relationship_type_counts"],
                model=model,
            )

        types_path = save_discovered_types(
            consolidated,
            merged["entity_type_counts"],
            merged["relationship_type_counts"],
            consolidation_method=consolidation_strategy,
        )
        merged["consolidated_types"] = {
            "entity_types": consolidated.entity_types,
            "relationship_types": consolidated.relationship_types,
            "rationale": consolidated.rationale,
        }
        merged["types_path"] = str(types_path)

        logger.info(
            f"Consolidated to {len(consolidated.entity_types)} entity types, "
            f"{len(consolidated.relationship_types)} relationship types"
        )

    merged["extraction_path"] = str(extraction_path)
    merged["extractions_dir"] = str(extractions_dir)
    merged["processed_books"] = processed
    merged["skipped_books"] = skipped

    return merged


def reconsolidate(
    strategy: str = "stratified",
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> Dict[str, Any]:
    """Re-run consolidation on existing extractions without re-extracting.

    Args:
        strategy: "stratified" (default) or "global".
        model: LLM model for consolidation.

    Returns:
        Dict with consolidated types and stats.
    """
    extractions_dir = DIR_GRAPH_DATA / "extractions"
    if not extractions_dir.exists():
        raise FileNotFoundError(
            f"Extractions not found: {extractions_dir}. "
            "Run: python -m src.stages.run_stage_4_5_autotune"
        )

    merged = merge_extractions(extractions_dir)
    logger.info(f"Loaded {merged['stats']['total_entities']:,} entities")

    if strategy == "stratified":
        consolidated = consolidate_stratified(
            extractions_dir,
            merged["relationship_type_counts"],
            model=model,
        )
    else:
        consolidated = consolidate_global(
            merged["entity_type_counts"],
            merged["relationship_type_counts"],
            model=model,
        )

    types_path = save_discovered_types(
        consolidated,
        merged["entity_type_counts"],
        merged["relationship_type_counts"],
        consolidation_method=strategy,
    )

    return {
        "consolidated_types": {
            "entity_types": consolidated.entity_types,
            "relationship_types": consolidated.relationship_types,
            "rationale": consolidated.rationale,
        },
        "types_path": str(types_path),
        "strategy": strategy,
        "stats": merged["stats"],
    }
