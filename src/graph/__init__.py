"""GraphRAG module for knowledge graph construction and retrieval.

## RAG Theory: GraphRAG (arXiv:2404.16130)

GraphRAG augments RAG with knowledge graph structures for:
- Cross-document relationship discovery (entity linking)
- Hierarchical community detection (Leiden algorithm)
- Global query answering via community summaries

While vector search excels at local queries ("What does X say about Y?"),
GraphRAG enables global queries ("What are the main themes across all documents?").

## Module Structure

- schemas.py: Pydantic models for entities, relationships, communities
- extractor.py: LLM-based entity/relationship extraction
- neo4j_client.py: Neo4j connection and Cypher operations
- community.py: Leiden detection + community summarization
- query.py: Graph retrieval strategy for hybrid search

## Data Flow

1. Section chunks → Entity extraction → Neo4j upload
2. Neo4j graph → Leiden communities → Community summaries (stored in Weaviate)
3. Query → LLM entity extraction → Graph traversal → Chunk ID discovery
4. Vector search (Weaviate) + Fetch graph-only chunks → RRF merge → Answer
"""

from .schemas import (
    GraphEntity,
    GraphRelationship,
    ExtractionResult,
    Community,
    CommunityMember,
)
from .extractor import (
    load_chunks_for_extraction,
    save_extraction_results,
)
from .neo4j_client import (
    get_driver,
    get_gds_client,
    verify_connection,
    upload_extraction_results,
    get_graph_stats,
)
from .community import (
    detect_and_summarize_communities,
    save_communities,
    load_communities,
)
from .query import (
    get_graph_chunk_ids,
    retrieve_graph_context,
    retrieve_community_context,
    format_graph_context_for_generation,
    fetch_chunks_by_ids,
    hybrid_graph_retrieval,
)

__all__ = [
    # Schemas
    "GraphEntity",
    "GraphRelationship",
    "ExtractionResult",
    "Community",
    "CommunityMember",
    # Extraction helpers
    "load_chunks_for_extraction",
    "save_extraction_results",
    # Neo4j
    "get_driver",
    "get_gds_client",
    "verify_connection",
    "upload_extraction_results",
    "get_graph_stats",
    # Community
    "detect_and_summarize_communities",
    "save_communities",
    "load_communities",
    # Query
    "get_graph_chunk_ids",
    "retrieve_graph_context",
    "retrieve_community_context",
    "format_graph_context_for_generation",
    "fetch_chunks_by_ids",
    "hybrid_graph_retrieval",
]
