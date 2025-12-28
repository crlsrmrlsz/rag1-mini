# GraphRAG: Knowledge Graph + Communities

> **Paper:** [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) | Edge et al. (Microsoft Research) | April 2024

Builds a knowledge graph from entities and relationships in the corpus, then uses Leiden community detection to create hierarchical summaries. Enables cross-document synthesis for "big picture" questions.

## TL;DR

GraphRAG extracts entities (people, concepts, brain regions) and relationships from each chunk, builds a knowledge graph in Neo4j, runs Leiden community detection to find clusters of related entities, and generates LLM summaries for each community. At query time, it combines vector search with community-based retrieval via RRF.

## Key Results (Paper)

- **72-83% win rate** on comprehensiveness vs baseline RAG
- **62-82% win rate** on diversity of answers
- **97% fewer tokens** at query time using community summaries

## The Problem

Vector search excels at finding specific facts but fails on "global" questions:

```
Query: "What are the main themes across all 19 books?"
```

No single chunk contains this answer. Traditional RAG would retrieve random chunks that happen to match keywords.

## The Solution

### Two-Phase Architecture

```
═══════════════════════════════════════════════════════════════
                    INDEXING PHASE (Offline)
═══════════════════════════════════════════════════════════════

[Source Documents]
        │
        ▼
[Text Chunks] (section chunker output)
        │
        ▼
[Entity & Relationship Extraction] ← LLM with structured output
        │
        ▼
[Knowledge Graph Construction] ← Neo4j
        │
        ▼
[Community Detection] ← Leiden Algorithm (hierarchical)
        │
        ▼
[Community Summarization] ← LLM generates report-like summaries

═══════════════════════════════════════════════════════════════
                    QUERY PHASE (Online)
═══════════════════════════════════════════════════════════════

[User Query]
        │
        ▼
[Extract Query Entities]
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ PARALLEL RETRIEVAL                                        │
│                                                           │
│  Vector Search → top-k chunks by embedding similarity     │
│  Community Search → relevant communities by entity match  │
└───────────────────────────────────────────────────────────┘
        │
        ▼
[RRF Merge]
        │
        ▼
[Generation with mixed context: chunks + community summaries]
```

### Key Components

**1. Entity Extraction**: LLM identifies named entities and relationships from each chunk

**2. Knowledge Graph**: Neo4j stores entities as nodes, relationships as edges

**3. Leiden Communities**: Groups of densely connected entities (Leiden is improved Louvain)

**4. Community Summaries**: LLM-generated descriptions of what each community represents

## Implementation Details

### Entity Extraction

```python
# src/graph/extractor.py

def extract_from_chunk(chunk: Dict, model: str) -> ExtractionResult:
    """Extract entities and relationships from a chunk."""

    prompt = GRAPHRAG_EXTRACTION_PROMPT.format(
        entity_types=", ".join(GRAPHRAG_ENTITY_TYPES),
        relationship_types=", ".join(GRAPHRAG_RELATIONSHIP_TYPES),
        text=chunk["text"],
    )

    # Use structured output for reliable JSON parsing
    result = call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=ExtractionResult,
        temperature=0.0,  # Deterministic
    )

    return result
```

### Entity Types (Domain-Specific)

```python
# src/config.py

GRAPHRAG_ENTITY_TYPES = [
    # Neuroscience
    "BRAIN_REGION",      # Prefrontal cortex, amygdala
    "NEUROTRANSMITTER",  # Dopamine, serotonin
    "RESEARCHER",        # Sapolsky, Kahneman

    # Philosophy
    "PHILOSOPHER",       # Marcus Aurelius, Seneca
    "CONCEPT",           # Virtue, hedonic adaptation
    "PRACTICE",          # Meditation, journaling

    # Cross-domain
    "BOOK", "THEORY", "PRINCIPLE"
]

GRAPHRAG_RELATIONSHIP_TYPES = [
    "AUTHORED", "RESEARCHES", "RELATES_TO",
    "INFLUENCES", "TEACHES", "SUPPORTS"
]
```

### Neo4j Schema

```cypher
// Nodes
(:Entity {
  name: String,
  type: String,
  description: String,
  chunk_ids: [String],
  mention_count: Integer
})

// Relationships
(:Entity)-[:RELATED_TO {
  description: String,
  strength: Integer,
  source_chunk_id: String
}]->(:Entity)

// Community assignments (from Leiden)
(:Entity)-[:BELONGS_TO]->(:Community)
```

### Leiden Community Detection

```python
# src/graph/community.py

def project_graph(gds: GraphDataScience, graph_name: str) -> Any:
    """Create GDS graph projection for community detection."""
    graph, result = gds.graph.project(
        graph_name,
        "Entity",
        {"RELATED_TO": {"orientation": "UNDIRECTED"}}
    )
    return graph

def detect_communities(gds, graph) -> List[Dict]:
    """Run Leiden algorithm."""
    result = gds.leiden.stream(
        graph,
        maxLevels=GRAPHRAG_LEIDEN_MAX_LEVELS,
        gamma=GRAPHRAG_LEIDEN_RESOLUTION,
    )
    return result.to_dict("records")
```

### Community Summarization

```python
def summarize_community(members: List[Dict], model: str) -> str:
    """Generate LLM summary for a community."""

    # Build context from member entities
    context = "\n".join([
        f"- {m['name']} ({m['type']}): {m['description']}"
        for m in members
    ])

    prompt = GRAPHRAG_COMMUNITY_PROMPT.format(
        entity_list=context,
        relationship_summary=...,
    )

    return call_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=GRAPHRAG_MAX_SUMMARY_TOKENS,
    )
```

### Query-Time Hybrid Retrieval

```python
# src/graph/query.py

def hybrid_graph_retrieval(
    query: str,
    preprocessed: PreprocessedQuery,
    weaviate_client,
    neo4j_driver,
) -> RRFResult:
    """Combine vector search with graph-based retrieval."""

    # 1. Vector search (standard)
    vector_results = weaviate_client.query_hybrid(query, ...)

    # 2. Extract entities from query
    query_entities = extract_query_entities(query, neo4j_driver)

    # 3. Find relevant communities
    communities = find_communities_for_entities(query_entities, neo4j_driver)

    # 4. Get community summaries as additional context
    community_context = [c.summary for c in communities]

    # 5. RRF merge
    return reciprocal_rank_fusion(
        [vector_results, community_results],
        ["vector", "community"],
    )
```

### Design Decisions

**Why Neo4j (not in-memory graph)?**
- GDS plugin provides Leiden algorithm
- Persistent storage for large graphs
- Cypher queries for flexible traversal
- Production-ready scaling

**Why Leiden (not Louvain)?**
- Guarantees connected communities (Louvain can produce disconnected ones)
- Better resolution parameter behavior
- Standard in GraphRAG implementations

**Why structured output for extraction?**
- JSON schema enforces consistent format
- Pydantic validation catches errors
- Reduces parsing failures

## When to Use

**Good for:**
- "Big picture" questions across documents
- Entity-relationship queries ("How does X relate to Y?")
- Theme and pattern discovery
- Cross-book synthesis

**Limitations:**
- Complex setup (Neo4j + GDS plugin)
- Higher indexing cost (entity extraction per chunk)
- Latency from Neo4j queries
- Overkill for simple factual queries

## Cost Analysis

For 19 books:
- **Entity extraction**: ~5,000 chunks × 1 LLM call = ~$3-5
- **Community summarization**: ~100 communities × 1 LLM call = ~$0.50
- **Storage**: Neo4j container (~500MB RAM)

## Pipeline Commands

```bash
# 1. Extract entities (after section chunking)
python -m src.stages.run_stage_4_6_graph_extract --strategy section

# 2. Upload to Neo4j + run Leiden
docker compose up -d neo4j
python -m src.stages.run_stage_6b_neo4j

# 3. Query with graphrag strategy
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag
```

## Results

See [Evaluation Results](../evaluation/results.md) for RAGAS metrics comparing GraphRAG against none, HyDE, and decomposition strategies.

## Related

- [RAPTOR](../chunking/raptor.md) — Alternative hierarchy via clustering
- [Query Decomposition](query-decomposition.md) — Alternative for complex queries
- [Auto-Tuning](../../memory-bank/graphrag-improvements.md) — Entity type discovery
