"""Neo4j client for GraphRAG knowledge graph operations.

## RAG Theory: Graph Database for Knowledge Graphs

Neo4j stores the extracted knowledge graph with:
- Nodes: Entities with labels (entity_type) and properties
- Relationships: Typed edges with properties (weight, description)
- Indexes: For fast entity lookup by normalized_name

The graph structure enables:
- Entity resolution (MERGE nodes by normalized_name)
- Relationship discovery (multi-hop traversal)
- Community detection (Leiden algorithm via GDS)

## Library Usage

Uses the official neo4j Python driver (v6):
- GraphDatabase.driver() for connection pooling
- driver.execute_query() for single-statement queries
- Session for multi-statement transactions
- GDS client for graph algorithms (Leiden, PageRank)

## Data Flow

1. Extraction results → MERGE entities → MERGE relationships
2. Create GDS graph projection → Run Leiden → Store community IDs
3. Query time: Match entities → Traverse → Return subgraph
"""

from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session
from graphdatascience import GraphDataScience

from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    GRAPHRAG_LEIDEN_RESOLUTION,
    GRAPHRAG_LEIDEN_MAX_LEVELS,
    GRAPHRAG_MIN_COMMUNITY_SIZE,
)
from src.shared.files import setup_logging

logger = setup_logging(__name__)


def get_driver() -> Driver:
    """Create and return a Neo4j driver instance.

    Uses connection parameters from config. The driver manages
    connection pooling and should be closed when done.

    Returns:
        Neo4j Driver instance.

    Raises:
        neo4j.exceptions.ServiceUnavailable: If Neo4j is not running.
        neo4j.exceptions.AuthError: If credentials are wrong.

    Example:
        >>> driver = get_driver()
        >>> driver.verify_connectivity()
        >>> driver.close()
    """
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    return driver


def get_gds_client(driver: Driver) -> GraphDataScience:
    """Create GDS client from existing driver.

    The GDS client provides Python API for Neo4j Graph Data Science
    algorithms like Leiden, PageRank, etc.

    Args:
        driver: Neo4j driver instance.

    Returns:
        GraphDataScience client.

    Example:
        >>> driver = get_driver()
        >>> gds = get_gds_client(driver)
        >>> gds.version()
        '2.x.x'
    """
    return GraphDataScience.from_neo4j_driver(driver)


@contextmanager
def neo4j_session(driver: Driver):
    """Context manager for Neo4j session.

    Ensures session is properly closed after use.

    Args:
        driver: Neo4j driver instance.

    Yields:
        Neo4j Session.

    Example:
        >>> with neo4j_session(driver) as session:
        ...     result = session.run("MATCH (n) RETURN count(n)")
    """
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def verify_connection(driver: Driver) -> bool:
    """Verify Neo4j connection is working.

    Args:
        driver: Neo4j driver instance.

    Returns:
        True if connection is successful.

    Raises:
        Exception: If connection fails.
    """
    driver.verify_connectivity()
    logger.info(f"Connected to Neo4j at {NEO4J_URI}")
    return True


def create_indexes(driver: Driver) -> None:
    """Create indexes for efficient entity lookup.

    Creates:
    - Index on Entity.normalized_name for MERGE operations
    - Index on Entity.entity_type for filtering

    Args:
        driver: Neo4j driver instance.
    """
    queries = [
        # Index for entity lookup by normalized name
        """
        CREATE INDEX entity_name IF NOT EXISTS
        FOR (e:Entity)
        ON (e.normalized_name)
        """,
        # Index for filtering by entity type
        """
        CREATE INDEX entity_type IF NOT EXISTS
        FOR (e:Entity)
        ON (e.entity_type)
        """,
        # Index for chunk source tracking
        """
        CREATE INDEX entity_chunk IF NOT EXISTS
        FOR (e:Entity)
        ON (e.source_chunk_id)
        """,
    ]

    for query in queries:
        driver.execute_query(query)

    logger.info("Created Neo4j indexes for Entity nodes")


def clear_graph(driver: Driver) -> int:
    """Delete all nodes and relationships in the graph.

    WARNING: Destructive operation. Use for testing or full rebuild.

    Args:
        driver: Neo4j driver instance.

    Returns:
        Number of nodes deleted.
    """
    result = driver.execute_query(
        "MATCH (n) DETACH DELETE n RETURN count(n) as deleted"
    )
    deleted = result.records[0]["deleted"]
    logger.info(f"Cleared graph: {deleted} nodes deleted")
    return deleted


def upload_entities(
    driver: Driver,
    entities: List[Dict[str, Any]],
    batch_size: int = 100,
) -> int:
    """Upload entities to Neo4j with Python-computed normalization.

    Uses MERGE to handle duplicates (same normalized_name = same node).
    Normalization is computed in Python for better deduplication
    (Unicode, stopwords, punctuation handling).

    Args:
        driver: Neo4j driver instance.
        entities: List of entity dicts from extraction.
        batch_size: Number of entities per transaction.

    Returns:
        Number of entities created/merged.

    Example:
        >>> entities = [{"name": "dopamine", "entity_type": "NEUROTRANSMITTER", ...}]
        >>> count = upload_entities(driver, entities)
        >>> print(count)
        1
    """
    from src.graph.schemas import GraphEntity

    # Pre-compute normalized names in Python (better than Cypher toLower/trim)
    for entity in entities:
        ge = GraphEntity(
            name=entity["name"],
            entity_type=entity.get("entity_type", ""),
        )
        entity["normalized_name"] = ge.normalized_name()

    total = 0

    # Process in batches
    for i in range(0, len(entities), batch_size):
        batch = entities[i : i + batch_size]

        # UNWIND batch for efficient multi-row insert
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {normalized_name: entity.normalized_name})
        ON CREATE SET
            e.name = entity.name,
            e.entity_type = entity.entity_type,
            e.description = entity.description,
            e.source_chunk_id = entity.source_chunk_id,
            e.created_at = datetime()
        ON MATCH SET
            e.description = CASE
                WHEN size(entity.description) > size(coalesce(e.description, ''))
                THEN entity.description
                ELSE e.description
            END
        RETURN count(e) as count
        """

        result = driver.execute_query(query, entities=batch)
        batch_count = result.records[0]["count"]
        total += batch_count

    logger.info(f"Uploaded {total} entities to Neo4j")
    return total


def upload_relationships(
    driver: Driver,
    relationships: List[Dict[str, Any]],
    batch_size: int = 100,
) -> int:
    """Upload relationships to Neo4j with Python-computed normalization.

    Matches entities by normalized_name and creates relationships.
    Uses MERGE to avoid duplicate relationships.

    Args:
        driver: Neo4j driver instance.
        relationships: List of relationship dicts from extraction.
        batch_size: Number of relationships per transaction.

    Returns:
        Number of relationships created/merged.

    Example:
        >>> rels = [{"source_entity": "dopamine", "target_entity": "reward", ...}]
        >>> count = upload_relationships(driver, rels)
        >>> print(count)
        1
    """
    from src.graph.schemas import GraphEntity

    # Pre-compute normalized names for source and target entities
    for rel in relationships:
        source_ge = GraphEntity(name=rel["source_entity"], entity_type="")
        target_ge = GraphEntity(name=rel["target_entity"], entity_type="")
        rel["source_normalized"] = source_ge.normalized_name()
        rel["target_normalized"] = target_ge.normalized_name()

    total = 0

    for i in range(0, len(relationships), batch_size):
        batch = relationships[i : i + batch_size]

        query = """
        UNWIND $rels AS rel
        MATCH (source:Entity {normalized_name: rel.source_normalized})
        MATCH (target:Entity {normalized_name: rel.target_normalized})
        MERGE (source)-[r:RELATED_TO {type: rel.relationship_type}]->(target)
        ON CREATE SET
            r.description = rel.description,
            r.weight = rel.weight,
            r.source_chunk_id = rel.source_chunk_id,
            r.created_at = datetime()
        RETURN count(r) as count
        """

        result = driver.execute_query(query, rels=batch)
        batch_count = result.records[0]["count"]
        total += batch_count

    logger.info(f"Uploaded {total} relationships to Neo4j")
    return total


def upload_extraction_results(
    driver: Driver,
    results: Dict[str, Any],
) -> Dict[str, int]:
    """Upload extraction results (entities + relationships) to Neo4j.

    Main entry point for graph construction. Creates indexes first,
    then uploads entities, then relationships.

    Args:
        driver: Neo4j driver instance.
        results: Dict from extract_from_chunks() with entities and relationships.

    Returns:
        Dict with entity_count and relationship_count.

    Example:
        >>> results = run_extraction(...)
        >>> counts = upload_extraction_results(driver, results)
        >>> print(counts)
        {"entity_count": 150, "relationship_count": 120}
    """
    # Create indexes first
    create_indexes(driver)

    # Upload entities
    entity_count = upload_entities(driver, results["entities"])

    # Upload relationships
    relationship_count = upload_relationships(driver, results["relationships"])

    return {
        "entity_count": entity_count,
        "relationship_count": relationship_count,
    }


def get_graph_stats(driver: Driver) -> Dict[str, Any]:
    """Get statistics about the knowledge graph.

    Args:
        driver: Neo4j driver instance.

    Returns:
        Dict with node_count, relationship_count, entity_types, etc.
    """
    # Node count
    node_result = driver.execute_query("MATCH (n:Entity) RETURN count(n) as count")
    node_count = node_result.records[0]["count"]

    # Relationship count
    rel_result = driver.execute_query("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count")
    rel_count = rel_result.records[0]["count"]

    # Entity types breakdown
    type_result = driver.execute_query("""
        MATCH (e:Entity)
        RETURN e.entity_type as type, count(e) as count
        ORDER BY count DESC
    """)
    entity_types = {r["type"]: r["count"] for r in type_result.records}

    # Relationship types breakdown
    rel_type_result = driver.execute_query("""
        MATCH ()-[r:RELATED_TO]->()
        RETURN r.type as type, count(r) as count
        ORDER BY count DESC
    """)
    relationship_types = {r["type"]: r["count"] for r in rel_type_result.records}

    return {
        "node_count": node_count,
        "relationship_count": rel_count,
        "entity_types": entity_types,
        "relationship_types": relationship_types,
    }


def find_entity_neighbors(
    driver: Driver,
    entity_name: str,
    max_hops: int = 2,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Find entities connected to a given entity within N hops.

    Used for local graph retrieval during query processing.

    Args:
        driver: Neo4j driver instance.
        entity_name: Name of the entity to start from.
        max_hops: Maximum number of relationship hops (default 2).
        limit: Maximum number of results.

    Returns:
        List of connected entities with relationship info.

    Example:
        >>> neighbors = find_entity_neighbors(driver, "dopamine", max_hops=2)
        >>> for n in neighbors:
        ...     print(n["name"], n["path_length"])
    """
    query = f"""
    MATCH (start:Entity {{normalized_name: toLower(trim($entity_name))}})
    MATCH path = (start)-[*1..{max_hops}]-(neighbor:Entity)
    WHERE start <> neighbor
    RETURN DISTINCT
        neighbor.name as name,
        neighbor.entity_type as entity_type,
        neighbor.description as description,
        length(path) as path_length
    ORDER BY path_length, name
    LIMIT $limit
    """

    result = driver.execute_query(query, entity_name=entity_name, limit=limit)
    return [dict(r) for r in result.records]


def find_entities_by_names(
    driver: Driver,
    entity_names: List[str],
) -> List[Dict[str, Any]]:
    """Find entities by a list of names.

    Used to locate entities mentioned in a query.

    Args:
        driver: Neo4j driver instance.
        entity_names: List of entity names to search for.

    Returns:
        List of matching entity dicts.
    """
    query = """
    UNWIND $names AS name
    MATCH (e:Entity)
    WHERE toLower(trim(e.name)) = toLower(trim(name))
       OR toLower(trim(e.normalized_name)) = toLower(trim(name))
    RETURN DISTINCT
        e.name as name,
        e.entity_type as entity_type,
        e.description as description,
        e.source_chunk_id as source_chunk_id
    """

    result = driver.execute_query(query, names=entity_names)
    return [dict(r) for r in result.records]
