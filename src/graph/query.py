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
import json

import numpy as np
from neo4j import Driver

from src.config import (
    GRAPHRAG_TOP_COMMUNITIES,
    GRAPHRAG_TRAVERSE_DEPTH,
    GRAPHRAG_EXTRACTION_MODEL,
    GRAPHRAG_ENTITY_TYPES,
    GRAPHRAG_QUERY_EXTRACTION_PROMPT,
    DIR_GRAPH_DATA,
)
from src.shared.files import setup_logging
from src.shared.openrouter_client import call_structured_completion
from src.rag_pipeline.embedding.embedder import embed_texts
from .neo4j_client import find_entity_neighbors, find_entities_by_names
from .community import load_communities
from .schemas import Community, QueryEntities

logger = setup_logging(__name__)


# ============================================================================
# LLM-based Query Entity Extraction
# ============================================================================

def _get_entity_types() -> List[str]:
    """Get entity types, preferring discovered types if available.

    Returns:
        List of entity type strings.
    """
    discovered_path = DIR_GRAPH_DATA / "discovered_types.json"
    if discovered_path.exists():
        with open(discovered_path, "r") as f:
            data = json.load(f)
        logger.debug(f"Using {len(data['consolidated_entity_types'])} discovered entity types")
        return data["consolidated_entity_types"]
    else:
        logger.debug("Using default entity types from config")
        return GRAPHRAG_ENTITY_TYPES


def extract_query_entities_llm(
    query: str,
    model: str = GRAPHRAG_EXTRACTION_MODEL,
) -> List[str]:
    """Extract entities from query using LLM.

    Uses structured output to identify entity mentions in the query,
    including lowercase conceptual terms that regex would miss.

    Args:
        query: User query string.
        model: OpenRouter model ID (default: claude-3-haiku).

    Returns:
        List of entity names extracted from query.

    Example:
        >>> extract_query_entities_llm("What creates lasting happiness?")
        ["happiness", "pleasure", "hedonic adaptation"]
    """
    entity_types = _get_entity_types()

    prompt = GRAPHRAG_QUERY_EXTRACTION_PROMPT.format(
        entity_types=", ".join(entity_types),
        query=query,
    )

    try:
        result = call_structured_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            response_model=QueryEntities,
            temperature=0.0,
            max_tokens=500,
        )
        entities = [e.name for e in result.entities]
        logger.info(f"LLM extracted entities: {entities}")
        return entities
    except Exception as e:
        logger.warning(f"LLM query extraction failed: {e}")
        return []


# ============================================================================
# Main Entity Extraction Function
# ============================================================================


def extract_query_entities(
    query: str,
    driver: Optional[Driver] = None,
    use_llm: bool = True,
) -> List[str]:
    """Extract entity mentions from query using LLM + Neo4j validation.

    Primary method: LLM-based extraction (handles conceptual terms)
    Fallback: Regex for capitalized words (if LLM fails)
    Validation: Neo4j lookup to verify entities exist in graph

    Args:
        query: User query string.
        driver: Optional Neo4j driver for entity lookup.
        use_llm: Whether to use LLM extraction (default True).

    Returns:
        List of entity names found in query.

    Example:
        >>> extract_query_entities("What creates lasting happiness?")
        ["happiness", "pleasure", "hedonic adaptation"]
        >>> extract_query_entities("How does Sapolsky explain stress?")
        ["Sapolsky", "stress"]
    """
    entities = []

    # Primary: LLM-based extraction
    if use_llm:
        entities = extract_query_entities_llm(query)

    # Fallback: Regex for capitalized words
    if not entities:
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        capitalized = re.findall(cap_pattern, query)
        entities.extend(capitalized)
        if entities:
            logger.info(f"Regex fallback entities: {entities}")

    # Validate against Neo4j if driver provided
    if driver and entities:
        db_entities = find_entities_by_names(driver, entities)
        validated = [e["name"] for e in db_entities]
        # Add validated entities (may have different casing)
        entities.extend(validated)

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


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score in range [-1, 1].
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_product = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm_product)


def retrieve_community_context(
    query: str,
    communities: Optional[List[Community]] = None,
    top_k: int = GRAPHRAG_TOP_COMMUNITIES,
) -> List[Dict[str, Any]]:
    """Retrieve relevant community summaries using embedding similarity.

    Uses cosine similarity between query embedding and community summary
    embeddings for semantic matching. Falls back to keyword matching if
    embeddings are not available.

    Args:
        query: User query string.
        communities: List of Community objects (loads from file if None).
        top_k: Number of top communities to return.

    Returns:
        List of community dicts with summary, member info, and score.

    Example:
        >>> context = retrieve_community_context("What are the main themes?")
        >>> for c in context:
        ...     print(c["summary"][:100], c["score"])
    """
    if communities is None:
        try:
            communities = load_communities()
        except FileNotFoundError:
            logger.warning("No communities file found, skipping community retrieval")
            return []

    if not communities:
        return []

    # Check if embeddings are available
    has_embeddings = any(c.embedding for c in communities)

    if has_embeddings:
        # Embedding-based retrieval (preferred)
        logger.debug("Using embedding-based community retrieval")
        query_embedding = embed_texts([query])[0]

        scored = []
        for community in communities:
            if community.embedding:
                similarity = cosine_similarity(query_embedding, community.embedding)
                scored.append((similarity, community))

        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        # Fallback: keyword matching (legacy)
        logger.debug("No community embeddings, using keyword fallback")
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

        scored.sort(key=lambda x: x[0], reverse=True)

    # Return top-k
    results = []
    for score, community in scored[:top_k]:
        results.append({
            "community_id": community.community_id,
            "summary": community.summary,
            "member_count": community.member_count,
            "score": float(score),
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


def hybrid_graph_retrieval(
    query: str,
    driver: Driver,
    vector_results: List[Dict[str, Any]],
    top_k: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge vector search results with graph traversal.

    Enhances vector search with knowledge graph context:
    1. Traverses graph from query entities to find related chunks
    2. Boosts vector results that also appear in graph traversal
    3. Adds community summaries for thematic context

    Args:
        query: User query string.
        driver: Neo4j driver instance.
        vector_results: Results from Weaviate vector search (list of dicts).
        top_k: Number of results to return.

    Returns:
        Tuple of:
        - Merged results (graph-boosted results first, then others)
        - Metadata dict with entities, graph context, and communities

    Example:
        >>> driver = get_driver()
        >>> results, meta = hybrid_graph_retrieval("What is dopamine?", driver, vector_results)
        >>> print(meta["query_entities"])
    """
    # Get graph chunk IDs and metadata
    graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)

    # Get community context for thematic enrichment
    community_context = retrieve_community_context(query)

    # Build metadata
    metadata = {
        "query_entities": graph_meta.get("query_entities", []),
        "graph_context": graph_meta.get("graph_context", []),
        "community_context": community_context,
        "graph_chunk_count": len(graph_chunk_ids),
    }

    if not graph_chunk_ids:
        # No graph results, return vector results as-is
        logger.info("No graph chunks found, returning vector results only")
        return vector_results[:top_k], metadata

    # Enrich vector results with graph boost flag
    enriched, graph_only_ids = enrich_results_with_graph(
        vector_results, graph_chunk_ids, community_context
    )

    # Sort: graph-boosted results first, then by original score
    # This is a simple boost strategy; could use RRF for more sophistication
    boosted = [r for r in enriched if r.get("graph_boost")]
    non_boosted = [r for r in enriched if not r.get("graph_boost")]

    # Combine: boosted first (preserving their order), then non-boosted
    merged = boosted + non_boosted

    logger.info(
        f"Hybrid retrieval: {len(boosted)} graph-boosted, "
        f"{len(non_boosted)} vector-only, {len(graph_only_ids)} graph-only (not fetched)"
    )

    metadata["boosted_count"] = len(boosted)
    metadata["graph_only_count"] = len(graph_only_ids)

    return merged[:top_k], metadata


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
