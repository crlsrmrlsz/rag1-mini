"""PageRank centrality computation for GraphRAG entity importance ranking.

## RAG Theory: Entity Importance in Communities

PageRank identifies "hub" entities that are well-connected to other
important entities. This is more informative than simple degree count
because it considers the quality of connections, not just quantity.

Used for:
1. Prioritizing entities in community summaries (top-k by PageRank)
2. Ordering members in generation context
3. Selecting representative entities for map-reduce

## Library Usage

Uses Neo4j GDS (Graph Data Science) library:
- gds.pageRank.stream() - Compute PageRank scores for all nodes
- gds.graph.project() - Create subgraph projection if needed

## Data Flow

1. After Leiden: Run PageRank on full graph projection
2. Store scores in Neo4j as `e.pagerank` property
3. Retrieve scores when building CommunityMember objects
4. Sort members by PageRank for summarization focus
"""

from typing import Any, Optional

from neo4j import Driver
from graphdatascience import GraphDataScience

from src.config import (
    GRAPHRAG_PAGERANK_DAMPING,
    GRAPHRAG_PAGERANK_ITERATIONS,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


def compute_pagerank(
    gds: GraphDataScience,
    graph: Any,
    damping_factor: float = GRAPHRAG_PAGERANK_DAMPING,
    max_iterations: int = GRAPHRAG_PAGERANK_ITERATIONS,
) -> dict[int, float]:
    """Compute PageRank centrality for all nodes in the graph.

    PageRank measures node importance based on the structure of
    incoming links. Higher scores indicate more influential nodes.

    Args:
        gds: GraphDataScience client instance.
        graph: GDS graph projection from project_graph().
        damping_factor: Probability of continuing walk (default 0.85).
        max_iterations: Maximum iterations for convergence (default 20).

    Returns:
        Dict mapping Neo4j internal node_id to PageRank score.

    Example:
        >>> scores = compute_pagerank(gds, graph)
        >>> top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        >>> for node_id, score in top_nodes:
        ...     print(f"Node {node_id}: {score:.4f}")
    """
    logger.info(
        f"Computing PageRank with damping={damping_factor}, "
        f"maxIterations={max_iterations}"
    )

    result = gds.pageRank.stream(
        graph,
        dampingFactor=damping_factor,
        maxIterations=max_iterations,
    )

    scores = {}
    for record in result.itertuples():
        scores[record.nodeId] = record.score

    # Log statistics
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        avg_score = sum(scores.values()) / len(scores)
        logger.info(
            f"PageRank computed for {len(scores)} nodes: "
            f"min={min_score:.4f}, max={max_score:.4f}, avg={avg_score:.4f}"
        )

    return scores


def write_pagerank_to_neo4j(
    driver: Driver,
    pagerank_scores: dict[int, float],
    batch_size: int = 1000,
) -> int:
    """Write PageRank scores to Neo4j Entity nodes.

    Stores the PageRank score as a property on each Entity node
    for later retrieval during community summarization.

    Args:
        driver: Neo4j driver instance.
        pagerank_scores: Dict from compute_pagerank().
        batch_size: Batch size for UNWIND operations.

    Returns:
        Number of nodes updated.

    Example:
        >>> scores = compute_pagerank(gds, graph)
        >>> count = write_pagerank_to_neo4j(driver, scores)
        >>> print(f"Updated {count} nodes with PageRank scores")
    """
    if not pagerank_scores:
        logger.warning("No PageRank scores to write")
        return 0

    # Convert to list of dicts for UNWIND
    assignments = [
        {"node_id": node_id, "score": score}
        for node_id, score in pagerank_scores.items()
    ]

    query = """
    UNWIND $assignments AS assignment
    MATCH (e:Entity)
    WHERE id(e) = assignment.node_id
    SET e.pagerank = assignment.score
    RETURN count(e) as count
    """

    total_updated = 0

    # Process in batches
    for i in range(0, len(assignments), batch_size):
        batch = assignments[i : i + batch_size]
        result = driver.execute_query(query, assignments=batch)
        count = result.records[0]["count"]
        total_updated += count

    logger.info(f"Wrote PageRank scores to {total_updated} Entity nodes")
    return total_updated


def get_pagerank_scores_from_neo4j(
    driver: Driver,
    node_ids: Optional[set[int]] = None,
) -> dict[int, float]:
    """Retrieve PageRank scores from Neo4j.

    Useful for loading scores without recomputing (e.g., after restart).

    Args:
        driver: Neo4j driver instance.
        node_ids: Optional set of node IDs to filter (None = all).

    Returns:
        Dict mapping node_id to PageRank score.
    """
    if node_ids:
        query = """
        MATCH (e:Entity)
        WHERE id(e) IN $node_ids AND e.pagerank IS NOT NULL
        RETURN id(e) as node_id, e.pagerank as score
        """
        result = driver.execute_query(query, node_ids=list(node_ids))
    else:
        query = """
        MATCH (e:Entity)
        WHERE e.pagerank IS NOT NULL
        RETURN id(e) as node_id, e.pagerank as score
        """
        result = driver.execute_query(query)

    scores = {r["node_id"]: r["score"] for r in result.records}
    logger.info(f"Retrieved {len(scores)} PageRank scores from Neo4j")
    return scores


def rank_entities_by_pagerank(
    pagerank_scores: dict[int, float],
    node_ids: set[int],
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Rank entities within a community by PageRank.

    Args:
        pagerank_scores: Full PageRank scores dict from compute_pagerank().
        node_ids: Set of node IDs in the community.
        top_k: Number of top entities to return.

    Returns:
        List of (node_id, score) tuples, sorted descending by score.

    Example:
        >>> community_nodes = {1, 2, 3, 4, 5}
        >>> top = rank_entities_by_pagerank(scores, community_nodes, top_k=3)
        >>> for node_id, score in top:
        ...     print(f"Node {node_id}: {score:.4f}")
    """
    # Filter to community nodes and get scores
    community_scores = [
        (node_id, pagerank_scores.get(node_id, 0.0))
        for node_id in node_ids
    ]

    # Sort by score descending
    community_scores.sort(key=lambda x: x[1], reverse=True)

    return community_scores[:top_k]


def normalize_pagerank_scores(
    scores: dict[int, float],
) -> dict[int, float]:
    """Normalize PageRank scores to [0, 1] range.

    Useful for comparison across different graph sizes.

    Args:
        scores: Raw PageRank scores.

    Returns:
        Normalized scores in [0, 1] range.
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    score_range = max_score - min_score

    if score_range == 0:
        # All scores equal, normalize to 0.5
        return {node_id: 0.5 for node_id in scores}

    return {
        node_id: (score - min_score) / score_range
        for node_id, score in scores.items()
    }
