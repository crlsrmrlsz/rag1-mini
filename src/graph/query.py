"""Graph retrieval strategy for hybrid GraphRAG + vector search.

## RAG Theory: Hybrid Retrieval in GraphRAG

GraphRAG combines two retrieval methods:
1. **Local search**: Entity matching → Graph traversal → Related chunks
2. **Global search**: Query → Community summary matching → Theme context

The hybrid approach uses RRF (Reciprocal Rank Fusion) to merge:
- Vector search results (semantic similarity)
- Graph traversal results (relationship-based)
- Community summaries (thematic context)

## Library Usage

Uses existing infrastructure:
- Neo4j for graph queries
- Weaviate for vector search
- RRF from src/rag_pipeline/retrieval/rrf.py

## Data Flow

1. Query → Extract entity mentions
2. Match entities in Neo4j → Traverse graph → Get related chunks
3. Vector search in Weaviate → Get similar chunks
4. RRF merge both result sets → Return top-k chunks
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import re

from neo4j import Driver

from src.config import (
    GRAPHRAG_TOP_COMMUNITIES,
    GRAPHRAG_TRAVERSE_DEPTH,
)
from src.shared.files import setup_logging
from .neo4j_client import find_entity_neighbors, find_entities_by_names
from .community import load_communities
from .schemas import Community

logger = setup_logging(__name__)


def extract_query_entities(
    query: str,
    driver: Optional[Driver] = None,
) -> List[str]:
    """Extract potential entity mentions from query.

    Uses pattern matching + optional Neo4j lookup:
    1. Capitalized words/phrases (proper nouns)
    2. Neo4j database lookup for known entities

    Note: Does not use hardcoded domain terms to maintain domain-agnostic design.

    Args:
        query: User query string.
        driver: Optional Neo4j driver for entity lookup.

    Returns:
        List of potential entity names.

    Example:
        >>> extract_query_entities("How does Sapolsky explain stress?")
        ["Sapolsky"]
    """
    entities = []

    # Pattern 1: Capitalized words (excluding sentence starts)
    # Match sequences of capitalized words (proper nouns, concepts)
    cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    capitalized = re.findall(cap_pattern, query)
    entities.extend(capitalized)

    # Pattern 2: If driver provided, check Neo4j for matches
    # This finds entities from the actual corpus graph
    if driver and entities:
        db_entities = find_entities_by_names(driver, entities)
        entities.extend([e["name"] for e in db_entities])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        e_lower = e.lower()
        if e_lower not in seen:
            seen.add(e_lower)
            unique.append(e)

    return unique


def retrieve_graph_context(
    query: str,
    driver: Driver,
    max_hops: int = GRAPHRAG_TRAVERSE_DEPTH,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Retrieve context from knowledge graph based on query.

    Extracts entities from query, traverses graph, returns
    related entities and their source chunks.

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        max_hops: Maximum traversal depth.
        limit: Maximum results per entity.

    Returns:
        List of dicts with entity info and source_chunk_id.

    Example:
        >>> context = retrieve_graph_context("What causes stress?", driver)
        >>> for c in context:
        ...     print(c["name"], c["source_chunk_id"])
    """
    # Extract entities from query
    query_entities = extract_query_entities(query, driver)

    if not query_entities:
        logger.debug("No entities found in query for graph traversal")
        return []

    logger.info(f"Graph traversal from entities: {query_entities}")

    # Traverse from each entity
    all_neighbors = []
    seen_names = set()

    for entity_name in query_entities:
        neighbors = find_entity_neighbors(
            driver, entity_name, max_hops=max_hops, limit=limit
        )
        for neighbor in neighbors:
            if neighbor["name"].lower() not in seen_names:
                seen_names.add(neighbor["name"].lower())
                all_neighbors.append(neighbor)

    logger.info(f"Graph traversal found {len(all_neighbors)} related entities")
    return all_neighbors


def get_chunk_ids_from_graph(
    graph_context: List[Dict[str, Any]],
) -> List[str]:
    """Extract unique chunk IDs from graph context.

    Used to fetch full chunk content from Weaviate or files.

    Args:
        graph_context: List from retrieve_graph_context().

    Returns:
        List of unique chunk IDs.
    """
    chunk_ids = set()
    for entity in graph_context:
        if entity.get("source_chunk_id"):
            chunk_ids.add(entity["source_chunk_id"])
    return list(chunk_ids)


def retrieve_community_context(
    query: str,
    communities: Optional[List[Community]] = None,
    top_k: int = GRAPHRAG_TOP_COMMUNITIES,
) -> List[Dict[str, Any]]:
    """Retrieve relevant community summaries for global queries.

    Uses simple keyword matching for now. Could be enhanced with
    embedding similarity on community summaries.

    Args:
        query: User query string.
        communities: List of Community objects (loads from file if None).
        top_k: Number of top communities to return.

    Returns:
        List of community dicts with summary and member info.

    Example:
        >>> context = retrieve_community_context("What are the main themes?")
        >>> for c in context:
        ...     print(c["summary"][:100])
    """
    if communities is None:
        try:
            communities = load_communities()
        except FileNotFoundError:
            logger.warning("No communities file found, skipping community retrieval")
            return []

    if not communities:
        return []

    # Score communities by keyword overlap with query
    query_words = set(query.lower().split())

    scored = []
    for community in communities:
        # Count keyword matches in summary
        summary_words = set(community.summary.lower().split())
        overlap = len(query_words & summary_words)

        # Also check member names
        member_names = " ".join(m.entity_name for m in community.members).lower()
        member_overlap = sum(1 for w in query_words if w in member_names)

        score = overlap + member_overlap * 2  # Weight member matches higher

        if score > 0:
            scored.append((score, community))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top-k
    results = []
    for score, community in scored[:top_k]:
        results.append({
            "community_id": community.community_id,
            "summary": community.summary,
            "member_count": community.member_count,
            "score": score,
        })

    logger.info(f"Community retrieval found {len(results)} relevant communities")
    return results


def get_graph_chunk_ids(
    query: str,
    driver: Driver,
) -> Tuple[List[str], Dict[str, Any]]:
    """Get chunk IDs from graph traversal for a query.

    Extracts entities from query, traverses graph, and returns
    source chunk IDs from related entities. These can be used
    to boost or add chunks to vector search results.

    Args:
        query: User query string.
        driver: Neo4j driver instance.

    Returns:
        Tuple of:
        - List of chunk IDs found via graph traversal
        - Metadata dict with query_entities and graph_context

    Raises:
        neo4j.exceptions.ServiceUnavailable: If Neo4j connection fails.

    Example:
        >>> chunk_ids, meta = get_graph_chunk_ids("What is dopamine?", driver)
        >>> print(chunk_ids[:3])
        ["behave::chunk_42", "behave::chunk_43", ...]
    """
    metadata = {
        "query_entities": [],
        "graph_context": [],
    }

    # Extract entities from query
    query_entities = extract_query_entities(query, driver)
    metadata["query_entities"] = query_entities

    if not query_entities:
        return [], metadata

    # Get graph context via traversal
    graph_context = retrieve_graph_context(query, driver)
    metadata["graph_context"] = graph_context

    # Extract unique chunk IDs
    chunk_ids = get_chunk_ids_from_graph(graph_context)

    logger.info(
        f"Graph retrieval: {len(query_entities)} entities -> "
        f"{len(graph_context)} neighbors -> {len(chunk_ids)} chunks"
    )

    return chunk_ids, metadata


def enrich_results_with_graph(
    vector_results: List[Dict[str, Any]],
    graph_chunk_ids: List[str],
    community_context: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Enrich vector results with graph-derived chunk IDs.

    Adds graph_boost flag to vector results that also appear in graph traversal.
    Returns set of graph-only chunk IDs not in vector results.

    Args:
        vector_results: List of chunks from vector search.
        graph_chunk_ids: Chunk IDs from graph traversal.
        community_context: Community summaries for context.

    Returns:
        Tuple of:
        - Enriched vector results with graph_boost flag
        - Set of graph-only chunk IDs (for potential fetch)

    Example:
        >>> enriched, extras = enrich_results_with_graph(results, graph_ids, [])
        >>> for r in enriched:
        ...     if r.get("graph_boost"):
        ...         print("Boosted:", r["chunk_id"])
    """
    # Create set of vector result chunk IDs for fast lookup
    vector_chunk_ids = {r.get("chunk_id") for r in vector_results if r.get("chunk_id")}
    graph_set = set(graph_chunk_ids)

    # Mark vector results that also appear in graph
    for result in vector_results:
        chunk_id = result.get("chunk_id")
        if chunk_id and chunk_id in graph_set:
            result["graph_boost"] = True

    # Find graph-only chunks not in vector results
    graph_only = graph_set - vector_chunk_ids

    return vector_results, graph_only


def format_graph_context_for_generation(
    metadata: Dict[str, Any],
    max_chars: int = 2000,
) -> str:
    """Format graph metadata as additional context for answer generation.

    Includes entity relationships and community summaries
    to augment the retrieved chunks.

    Args:
        metadata: Dict from hybrid_graph_retrieval().
        max_chars: Maximum characters for context.

    Returns:
        Formatted context string for LLM prompt.
    """
    lines = []

    # Add community summaries if available
    if metadata.get("community_context"):
        lines.append("## Relevant Themes (from document corpus)")
        for comm in metadata["community_context"][:2]:  # Top 2 communities
            lines.append(f"\n{comm['summary']}")

    # Add entity relationships if available
    if metadata.get("graph_context"):
        lines.append("\n## Related Concepts (from knowledge graph)")
        for entity in metadata["graph_context"][:10]:  # Top 10 entities
            if entity.get("description"):
                lines.append(f"- {entity['name']}: {entity['description']}")
            else:
                lines.append(f"- {entity['name']} ({entity.get('entity_type', 'concept')})")

    context = "\n".join(lines)

    # Truncate if needed
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[... truncated]"

    return context
