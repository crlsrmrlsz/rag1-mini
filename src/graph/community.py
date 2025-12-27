"""Leiden community detection and summarization for GraphRAG.

## RAG Theory: Community Detection in GraphRAG

GraphRAG uses the Leiden algorithm (improvement over Louvain) to:
1. Detect communities of related entities in the knowledge graph
2. Create hierarchical community structure (multi-level)
3. Generate LLM summaries for each community

These community summaries enable "global queries" that synthesize
information across multiple documents (e.g., "What are the main themes?").

## Library Usage

Uses Neo4j GDS (Graph Data Science) for Leiden:
- gds.graph.project() - Create in-memory graph projection
- gds.leiden.stream() - Run Leiden, get community assignments
- gds.pageRank.stream() - Compute node centrality for ranking

## Data Flow

1. Project graph → GDS in-memory graph
2. Run Leiden → Community assignments per node
3. For each community: Collect members → Generate LLM summary
4. Store summaries in Neo4j and/or JSON for retrieval
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from neo4j import Driver
from graphdatascience import GraphDataScience

from src.config import (
    GRAPHRAG_LEIDEN_RESOLUTION,
    GRAPHRAG_LEIDEN_MAX_LEVELS,
    GRAPHRAG_MIN_COMMUNITY_SIZE,
    GRAPHRAG_SUMMARY_MODEL,
    GRAPHRAG_COMMUNITY_PROMPT,
    GRAPHRAG_MAX_SUMMARY_TOKENS,
    GRAPHRAG_MAX_CONTEXT_TOKENS,
    DIR_GRAPH_DATA,
)
from src.shared.openrouter_client import call_chat_completion
from src.shared.files import setup_logging
from .schemas import Community, CommunityMember
from .neo4j_client import get_gds_client

logger = setup_logging(__name__)


def project_graph(gds: GraphDataScience, graph_name: str = "graphrag") -> Any:
    """Create GDS graph projection for community detection.

    Projects Entity nodes and RELATED_TO relationships into
    GDS in-memory format for algorithm execution.

    Args:
        gds: GraphDataScience client instance.
        graph_name: Name for the projected graph.

    Returns:
        GDS Graph object.

    Raises:
        Exception: If projection fails (e.g., no data).
    """
    # Drop existing projection if exists
    if gds.graph.exists(graph_name).exists:
        gds.graph.drop(graph_name)
        logger.info(f"Dropped existing graph projection: {graph_name}")

    # Project the graph
    # Using native projection for Entity nodes and RELATED_TO relationships
    # Note: GDS only supports numeric properties, so we don't project entity_type (string)
    graph, result = gds.graph.project(
        graph_name,
        "Entity",  # Node label
        {
            "RELATED_TO": {
                "orientation": "UNDIRECTED",  # Leiden works on undirected
            }
        },
    )

    logger.info(
        f"Projected graph '{graph_name}': "
        f"{result['nodeCount']} nodes, {result['relationshipCount']} relationships"
    )

    return graph


def run_leiden(
    gds: GraphDataScience,
    graph: Any,
    resolution: float = GRAPHRAG_LEIDEN_RESOLUTION,
    max_levels: int = GRAPHRAG_LEIDEN_MAX_LEVELS,
) -> Dict[str, Any]:
    """Run Leiden community detection algorithm.

    Leiden improves on Louvain by guaranteeing well-connected communities.
    Returns hierarchical community assignments.

    Args:
        gds: GraphDataScience client instance.
        graph: GDS Graph object from project_graph().
        resolution: Higher = more, smaller communities (default 1.0).
        max_levels: Maximum hierarchy depth.

    Returns:
        Dict with:
        - community_count: Number of communities found
        - levels: Number of hierarchy levels
        - node_communities: List of (node_id, community_id) tuples

    Example:
        >>> result = run_leiden(gds, graph)
        >>> print(result["community_count"])
        12
    """
    # Run Leiden in stream mode to get results
    result = gds.leiden.stream(
        graph,
        gamma=resolution,  # Resolution parameter
        maxLevels=max_levels,
        includeIntermediateCommunities=True,  # Get hierarchy
    )

    # Convert to list of dicts
    node_communities = []
    for record in result.itertuples():
        node_communities.append({
            "node_id": record.nodeId,
            "community_id": record.communityId,
            "intermediate_ids": list(record.intermediateCommunityIds) if hasattr(record, 'intermediateCommunityIds') else [],
        })

    # Get unique community count
    unique_communities = set(nc["community_id"] for nc in node_communities)

    logger.info(
        f"Leiden found {len(unique_communities)} communities "
        f"across {len(node_communities)} nodes"
    )

    return {
        "community_count": len(unique_communities),
        "node_count": len(node_communities),
        "node_communities": node_communities,
    }


def write_communities_to_neo4j(
    driver: Driver,
    node_communities: List[Dict[str, Any]],
) -> int:
    """Write community assignments back to Neo4j nodes.

    Stores community_id as a property on each Entity node
    for later querying.

    Args:
        driver: Neo4j driver instance.
        node_communities: List from run_leiden() with node_id and community_id.

    Returns:
        Number of nodes updated.
    """
    query = """
    UNWIND $assignments AS assignment
    MATCH (e:Entity)
    WHERE id(e) = assignment.node_id
    SET e.community_id = assignment.community_id
    RETURN count(e) as count
    """

    result = driver.execute_query(query, assignments=node_communities)
    count = result.records[0]["count"]

    logger.info(f"Updated {count} nodes with community IDs")
    return count


def get_community_members(
    driver: Driver,
    community_id: int,
) -> List[CommunityMember]:
    """Get all entities belonging to a specific community.

    Args:
        driver: Neo4j driver instance.
        community_id: Community ID to query.

    Returns:
        List of CommunityMember objects.
    """
    query = """
    MATCH (e:Entity {community_id: $community_id})
    OPTIONAL MATCH (e)-[r:RELATED_TO]-()
    WITH e, count(r) as degree
    RETURN
        e.name as entity_name,
        e.entity_type as entity_type,
        e.description as description,
        degree
    ORDER BY degree DESC
    """

    result = driver.execute_query(query, community_id=community_id)

    members = []
    for record in result.records:
        members.append(CommunityMember(
            entity_name=record["entity_name"],
            entity_type=record["entity_type"] or "UNKNOWN",
            description=record["description"] or "",
            degree=record["degree"],
        ))

    return members


def get_community_relationships(
    driver: Driver,
    community_id: int,
) -> List[Dict[str, Any]]:
    """Get relationships within a community.

    Args:
        driver: Neo4j driver instance.
        community_id: Community ID to query.

    Returns:
        List of relationship dicts with source, target, type, description.
    """
    query = """
    MATCH (source:Entity {community_id: $community_id})-[r:RELATED_TO]->(target:Entity {community_id: $community_id})
    RETURN
        source.name as source,
        target.name as target,
        r.type as relationship_type,
        r.description as description
    """

    result = driver.execute_query(query, community_id=community_id)
    return [dict(r) for r in result.records]


def build_community_context(
    members: List[CommunityMember],
    relationships: List[Dict[str, Any]],
    max_tokens: int = GRAPHRAG_MAX_CONTEXT_TOKENS,
) -> str:
    """Build context string for community summarization.

    Formats entity and relationship information for LLM prompt.

    Args:
        members: List of community members.
        relationships: List of relationships within community.
        max_tokens: Approximate token limit (chars / 4).

    Returns:
        Formatted context string.
    """
    lines = []

    # Add entities
    lines.append("## Entities")
    for member in members:
        desc = f" - {member.description}" if member.description else ""
        lines.append(f"- {member.entity_name} ({member.entity_type}){desc}")

    # Add relationships
    if relationships:
        lines.append("\n## Relationships")
        for rel in relationships:
            desc = f": {rel['description']}" if rel.get("description") else ""
            lines.append(
                f"- {rel['source']} --[{rel['relationship_type']}]--> {rel['target']}{desc}"
            )

    context = "\n".join(lines)

    # Truncate if too long (approximate token limit)
    max_chars = max_tokens * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[... truncated]"

    return context


def summarize_community(
    members: List[CommunityMember],
    relationships: List[Dict[str, Any]],
    model: str = GRAPHRAG_SUMMARY_MODEL,
) -> str:
    """Generate LLM summary for a community.

    Uses community entities and relationships to generate
    a thematic summary for retrieval.

    Args:
        members: List of community members.
        relationships: List of relationships within community.
        model: LLM model for summarization.

    Returns:
        Summary string.
    """
    # Build context
    context = build_community_context(members, relationships)

    # Build prompt
    prompt = GRAPHRAG_COMMUNITY_PROMPT.format(community_context=context)

    # Call LLM
    messages = [{"role": "user", "content": prompt}]
    summary = call_chat_completion(
        messages=messages,
        model=model,
        temperature=0.3,
        max_tokens=GRAPHRAG_MAX_SUMMARY_TOKENS,
    )

    return summary.strip()


def get_community_ids_from_neo4j(driver: Driver) -> set:
    """Get unique community IDs already stored in Neo4j.

    Used for resume functionality when skipping Leiden.

    Args:
        driver: Neo4j driver instance.

    Returns:
        Set of community IDs.
    """
    query = """
    MATCH (e:Entity)
    WHERE e.community_id IS NOT NULL
    RETURN DISTINCT e.community_id as community_id
    """
    with driver.session() as session:
        result = session.run(query)
        return {record["community_id"] for record in result}


def detect_and_summarize_communities(
    driver: Driver,
    gds: GraphDataScience,
    min_size: int = GRAPHRAG_MIN_COMMUNITY_SIZE,
    model: str = GRAPHRAG_SUMMARY_MODEL,
    resume: bool = False,
    skip_leiden: bool = False,
) -> List[Community]:
    """Run full community detection and summarization pipeline.

    Main entry point for community processing:
    1. Project graph to GDS (unless skip_leiden)
    2. Run Leiden algorithm (unless skip_leiden)
    3. Write community IDs to Neo4j (unless skip_leiden)
    4. Generate summaries for each community (with resume support)

    Args:
        driver: Neo4j driver instance.
        gds: GraphDataScience client.
        min_size: Minimum community size to summarize.
        model: LLM model for summarization.
        resume: If True, load existing summaries and skip already-done communities.
        skip_leiden: If True, skip Leiden and use existing community_ids from Neo4j.

    Returns:
        List of Community objects with summaries.

    Example:
        >>> communities = detect_and_summarize_communities(driver, gds)
        >>> for c in communities:
        ...     print(c.community_id, c.member_count, c.summary[:50])
    """
    # Load existing summaries if resuming
    existing_summaries = {}
    if resume:
        try:
            existing = load_communities()
            existing_summaries = {c.community_id: c for c in existing}
            logger.info(f"Loaded {len(existing_summaries)} existing summaries for resume")
        except FileNotFoundError:
            logger.info("No existing summaries found, starting fresh")

    # Get community IDs
    if skip_leiden:
        # Use existing community IDs from Neo4j
        unique_ids = get_community_ids_from_neo4j(driver)
        logger.info(f"Loaded {len(unique_ids)} community IDs from Neo4j (skipping Leiden)")
        graph = None
    else:
        # Step 1: Project graph
        graph = project_graph(gds)

        # Step 2: Run Leiden
        leiden_result = run_leiden(gds, graph)

        # Step 3: Write community IDs to Neo4j
        write_communities_to_neo4j(driver, leiden_result["node_communities"])

        # Step 4: Get unique community IDs
        unique_ids = set(nc["community_id"] for nc in leiden_result["node_communities"])

    # Step 5: Process each community
    communities = list(existing_summaries.values()) if resume else []
    summarized_ids = set(existing_summaries.keys())
    new_summaries = 0

    for community_id in sorted(unique_ids):  # Sort for consistent ordering
        community_key = f"community_{community_id}"

        # Skip if already summarized (resume mode)
        if community_key in summarized_ids:
            continue

        # Get members
        members = get_community_members(driver, community_id)

        # Skip small communities
        if len(members) < min_size:
            continue

        # Get relationships
        relationships = get_community_relationships(driver, community_id)

        # Generate summary
        logger.info(
            f"Summarizing community {community_id} "
            f"({len(members)} members, {len(relationships)} relationships)"
        )
        summary = summarize_community(members, relationships, model=model)

        # Create Community object
        community = Community(
            community_id=community_key,
            level=0,  # Single level for now
            members=members,
            member_count=len(members),
            relationship_count=len(relationships),
            summary=summary,
        )
        communities.append(community)
        new_summaries += 1

        # Save incrementally after each summary
        save_communities(communities)

    # Cleanup: drop graph projection (if we created one)
    if graph is not None:
        gds.graph.drop(graph.name())

    logger.info(
        f"Generated {new_summaries} new community summaries "
        f"({len(communities)} total)"
    )
    return communities


def save_communities(
    communities: List[Community],
    output_name: str = "communities.json",
) -> Path:
    """Save community data to JSON file.

    Args:
        communities: List of Community objects.
        output_name: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = DIR_GRAPH_DATA
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name

    data = {
        "communities": [c.to_dict() for c in communities],
        "total_count": len(communities),
        "total_members": sum(c.member_count for c in communities),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(communities)} communities to {output_path}")
    return output_path


def load_communities(
    input_name: str = "communities.json",
) -> List[Community]:
    """Load communities from JSON file.

    Args:
        input_name: Input filename.

    Returns:
        List of Community objects.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    input_path = DIR_GRAPH_DATA / input_name

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    communities = []
    for c_data in data["communities"]:
        members = [CommunityMember(**m) for m in c_data.get("members", [])]
        community = Community(
            community_id=c_data["community_id"],
            level=c_data.get("level", 0),
            members=members,
            member_count=c_data["member_count"],
            relationship_count=c_data["relationship_count"],
            summary=c_data["summary"],
        )
        communities.append(community)

    logger.info(f"Loaded {len(communities)} communities from {input_path}")
    return communities
