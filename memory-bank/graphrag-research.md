# GraphRAG Research: Knowledge Graph-Augmented Retrieval for RAG

**Date:** 2025-12-28
**Status:** Implementation Complete
**Paper:** [From Local to Global: A Graph RAG Approach to Query-Focused Summarization (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130)
**Authors:** Darren Edge et al., Microsoft Research
**Published:** April 2024

---

## 1. Executive Summary

GraphRAG is Microsoft's graph-based approach to Retrieval-Augmented Generation that uses **LLM-generated knowledge graphs** combined with **hierarchical community summarization** to enable both local fact retrieval AND global sensemaking queries. Unlike traditional RAG which struggles with questions requiring synthesis across documents, GraphRAG creates a structured index that captures entities, relationships, and thematic clusters.

**Key Results:**
- **72-83% win rate** on comprehensiveness vs baseline RAG
- **62-82% win rate** on diversity of answers
- **97% fewer tokens** at query time using community summaries vs source text
- **+70% comprehensiveness** improvement (Microsoft claims)

**Why This Matters for RAG1-Mini:**
- Current approach fails on "global" questions like "What are the main themes across all books?"
- Graph structure captures relationships between concepts, authors, and ideas across documents
- Community summaries enable efficient query-time synthesis without reading all chunks
- Complements RAPTOR: RAPTOR = hierarchical document summaries; GraphRAG = cross-document relationships

---

## 2. How GraphRAG Works: The Core Architecture

### 2.1 The Two-Phase Approach

GraphRAG operates in two distinct phases:

```
+=======================================================================+
|                     INDEXING PHASE (Offline)                          |
+=======================================================================+
|                                                                       |
|  [Source Documents]                                                   |
|         |                                                             |
|         v                                                             |
|  [Text Chunks] (600 tokens, 100 overlap)                              |
|         |                                                             |
|         v                                                             |
|  [Entity & Relationship Extraction] <-- LLM with few-shot prompts     |
|         |                                                             |
|         v                                                             |
|  [Knowledge Graph Construction]                                       |
|         |                                                             |
|         v                                                             |
|  [Community Detection] <-- Leiden Algorithm (hierarchical)            |
|         |                                                             |
|         v                                                             |
|  [Community Summarization] <-- LLM generates report-like summaries    |
|                                                                       |
+=======================================================================+

+=======================================================================+
|                      QUERY PHASE (Online)                             |
+=======================================================================+
|                                                                       |
|  [User Query]                                                         |
|         |                                                             |
|         v                                                             |
|  [Select Community Summaries at appropriate level]                    |
|         |                                                             |
|         v                                                             |
|  [Map: Generate partial answers from each summary]                    |
|         |                                                             |
|         v                                                             |
|  [Reduce: Combine partial answers into final response]                |
|                                                                       |
+=======================================================================+
```

### 2.2 What Makes It Different from Traditional RAG

| Aspect | Traditional RAG | GraphRAG |
|--------|----------------|----------|
| **Index Structure** | Flat vector store of chunks | Knowledge graph + community hierarchy |
| **Query Type** | Works for specific facts | Works for global sensemaking |
| **Context** | Similar text chunks | Structured entity relationships |
| **Scalability** | Token cost grows with corpus | Community summaries compress context |
| **Relationships** | Implicit (via embeddings) | Explicit (edges in graph) |

### 2.3 Key Terminology

- **Entity**: A named concept, person, theory, brain region, etc. extracted from text
- **Relationship**: A typed connection between entities (e.g., "Marcus Aurelius" --AUTHORED--> "Meditations")
- **Claim/Covariate**: Verifiable factual statements attached to entities (e.g., "published 170 AD")
- **Community**: A cluster of closely related entities detected by Leiden algorithm
- **Community Summary**: LLM-generated report describing a community's themes and key facts

---

## 3. The GraphRAG Pipeline: Technical Deep Dive

### 3.1 Step 1: Text Chunking

**Configuration (from paper):**
- Chunk size: 600 tokens
- Overlap: 100 tokens
- Result: "Almost twice as many entity references when chunk size was 600 tokens than 2400"

**Rationale:** Smaller chunks = more focused entity extraction, better relationship precision.

**RAG1-Mini Adaptation:** Use our existing 800-token section chunks as base. Consider creating 400-token overlapping chunks specifically for graph extraction.

### 3.2 Step 2: Entity & Relationship Extraction

This is the core LLM-powered step that converts text to graph structure.

**What Gets Extracted:**

```json
{
  "entities": [
    {
      "name": "Dopamine",
      "type": "NEUROTRANSMITTER",
      "description": "A brain chemical involved in reward, motivation, and movement"
    },
    {
      "name": "Robert Sapolsky",
      "type": "PERSON",
      "description": "Stanford neuroscientist and author of Behave"
    }
  ],
  "relationships": [
    {
      "source": "Robert Sapolsky",
      "target": "Dopamine",
      "relation": "RESEARCHES",
      "description": "Sapolsky extensively studies dopamine's role in behavior",
      "strength": 8
    }
  ],
  "claims": [
    {
      "subject": "Robert Sapolsky",
      "claim": "Professor of biology at Stanford University",
      "evidence": "Based on author bio and book introduction"
    }
  ]
}
```

**Entity Types for RAG1-Mini (domain-specific):**

```python
ENTITY_TYPES = [
    # Neuroscience domain
    "BRAIN_REGION",      # Prefrontal cortex, amygdala, hippocampus
    "NEUROTRANSMITTER",  # Dopamine, serotonin, cortisol
    "COGNITIVE_PROCESS", # Memory, attention, decision-making
    "RESEARCHER",        # Sapolsky, Kahneman, Gazzaniga

    # Philosophy/Wisdom domain
    "PHILOSOPHER",       # Marcus Aurelius, Seneca, Buddha
    "PHILOSOPHICAL_SCHOOL", # Stoicism, Buddhism, Epicureanism
    "CONCEPT",           # Virtue, suffering, attachment
    "PRACTICE",          # Meditation, journaling, negative visualization

    # Cross-domain
    "BOOK",              # Behave, Meditations, Thinking Fast and Slow
    "THEORY",            # Dual process theory, Neuroplasticity
    "PRINCIPLE"          # Dichotomy of control, Hedonic adaptation
]

RELATIONSHIP_TYPES = [
    "AUTHORED",          # Person -> Book
    "RESEARCHES",        # Researcher -> Topic
    "RELATES_TO",        # Concept -> Concept
    "PART_OF",           # Brain region -> Brain system
    "INFLUENCES",        # Process -> Behavior
    "CONTRADICTS",       # Theory -> Theory
    "SUPPORTS",          # Evidence -> Claim
    "TEACHES",           # Philosopher -> Concept
    "PRACTICES",         # Practice -> Goal
]
```

**Self-Reflection for Quality:**

The paper uses an iterative extraction technique:
1. LLM extracts entities/relationships
2. LLM evaluates: "Are there any entities or relationships you missed?"
3. If YES, extract more; if NO, proceed
4. Uses logit bias to force yes/no response
5. Repeat up to N times (paper uses ~3 iterations)

**Prompt Template (adapted from paper):**

```python
ENTITY_EXTRACTION_PROMPT = """
You are an expert at extracting entities and relationships from text about
neuroscience, psychology, and philosophy.

Extract all entities and their relationships from the following text chunk.

For each entity, provide:
- name: The canonical name of the entity
- type: One of {entity_types}
- description: A brief description of what this entity is

For each relationship, provide:
- source: The source entity name
- target: The target entity name
- relation: One of {relationship_types}
- description: What this relationship represents
- strength: 1-10 indicating relationship importance

Text:
{text}

Respond in JSON format with "entities" and "relationships" arrays.
"""
```

### 3.3 Step 3: Knowledge Graph Construction

**Aggregation Process:**
1. Collect all entities from all chunks
2. Merge duplicate entities (same name = same node)
3. Link entities to source chunk IDs
4. Create edges for all relationships
5. Store entity descriptions and claims as node properties

**Neo4j Schema Design:**

```cypher
// Node types
(:Entity {
  name: String,
  type: String,
  description: String,
  chunk_ids: [String],  // Which chunks mention this entity
  mention_count: Integer
})

(:Chunk {
  chunk_id: String,
  book_id: String,
  text: String,
  embedding: [Float]  // For hybrid retrieval
})

// Relationship types
(:Entity)-[:RELATES_TO {
  type: String,
  description: String,
  strength: Integer,
  chunk_ids: [String]
}]->(:Entity)

(:Entity)-[:MENTIONED_IN]->(:Chunk)
(:Chunk)-[:FROM_BOOK]->(:Book)
```

**Entity Resolution:**

The paper uses simple string matching but notes "softer matching approaches can be used." For RAG1-Mini, consider:
- Exact match as baseline
- Embedding similarity for near-duplicates (e.g., "Marcus Aurelius" vs "Aurelius")
- LLM-based resolution for complex cases

### 3.4 Step 4: Community Detection with Leiden Algorithm

**What is Leiden?**

The Leiden algorithm is a hierarchical community detection algorithm that groups nodes with strong connections. Key advantages over Louvain:
- Guarantees well-connected communities
- Produces hierarchical structure (communities within communities)
- More efficient on large graphs

**How It Works:**

```
Phase 1: Local Moving
  - Each node starts in its own community
  - Nodes move to neighboring communities if it improves modularity
  - Repeat until no improvement

Phase 2: Refinement
  - Verify communities are internally well-connected
  - Split poorly-connected communities

Phase 3: Aggregation
  - Treat each community as a single super-node
  - Apply algorithm recursively
  - Build hierarchy of communities
```

**Hierarchical Levels:**

```
Level C0 (Root):    [  Community 1  ] [  Community 2  ]
                         /    \              |
Level C1:          [C1.1] [C1.2]         [C2.1]
                     /  \
Level C2:       [C1.1.1][C1.1.2]
                   ...
Level Cn (Leaves): Individual entities
```

**Paper Results:**
- Podcast dataset: 8,564 entities, 20,691 relationships
- News dataset: 15,754 entities, 19,520 relationships
- Community structure: 4-5 levels typical

**Implementation:**

```python
# Using graspologic (Microsoft's library)
from graspologic.partition import leiden

# Or using neo4j-gds (Graph Data Science library)
# CREATE GRAPH projection first, then:
CALL gds.leiden.stream('myGraph', {
  relationshipWeightProperty: 'strength',
  includeIntermediateCommunities: true,
  maxLevels: 10
})
YIELD nodeId, communityId, intermediateCommunityIds
```

**RAG1-Mini Considerations:**
- Smaller corpus = fewer communities (maybe 2-3 levels)
- Each book might form a natural high-level community
- Cross-book concepts (like "dopamine" or "Stoicism") will cluster

### 3.5 Step 5: Community Summarization

**Bottom-Up Summarization:**

```
For each level, bottom to top:
  For each community:
    If leaf community:
      Concatenate entity descriptions + relationships
      Truncate to context limit (8k tokens)
      Generate summary via LLM
    Else:
      If all sub-summaries fit:
        Summarize all sub-community summaries
      Else:
        Include as many sub-summaries as fit
        Summarize remaining
```

**Summary Format (from paper):**

```json
{
  "title": "Neuroscience of Decision Making",
  "summary": "This community focuses on the neural mechanisms underlying
             human decision-making, particularly the interplay between
             the prefrontal cortex and limbic system...",
  "rating": 8.5,
  "rating_explanation": "Highly relevant to understanding behavioral control",
  "findings": [
    {
      "summary": "The prefrontal cortex modulates emotional responses",
      "explanation": "Research shows PFC activity correlates with...",
      "data_references": ["chunk_123", "chunk_456"]
    }
  ]
}
```

**Token Efficiency:**
- C0 (root level): 9-43x fewer tokens than raw text
- C3 (detailed level): 26-33% fewer tokens than baseline

### 3.6 Step 6: Query Processing (Map-Reduce)

**Local Search (for specific queries):**
1. Extract entities from query
2. Find matching entities in graph
3. Expand to N-hop neighborhood
4. Retrieve associated chunks
5. Combine with vector similarity results
6. Generate answer

**Global Search (for sensemaking queries):**

```python
def global_search(query: str, community_level: int = 0):
    # 1. Get community summaries at specified level
    summaries = get_community_summaries(level=community_level)

    # 2. Shuffle and chunk summaries
    random.shuffle(summaries)
    chunks = chunk_summaries(summaries, max_tokens=8000)

    # 3. Map: Generate partial answers
    partial_answers = []
    for chunk in chunks:
        answer = llm.generate(
            f"Answer based on these community reports:\n{chunk}\n\nQuery: {query}"
        )
        score = llm.score_helpfulness(answer, query)  # 0-100
        if score > 0:
            partial_answers.append((answer, score))

    # 4. Sort by helpfulness
    partial_answers.sort(key=lambda x: x[1], reverse=True)

    # 5. Reduce: Combine into final answer
    context = "\n\n".join([a for a, s in partial_answers])
    final_answer = llm.generate(
        f"Synthesize these partial analyses:\n{context}\n\nQuery: {query}"
    )

    return final_answer
```

**Choosing Community Level:**
- **C0 (root)**: Best for very broad questions ("What are the main themes?")
- **C1-C2**: Good for topical questions ("How do Stoics view emotions?")
- **C3+ (detailed)**: Best for specific but multi-faceted questions

---

## 4. Neo4j Integration: Technical Details

### 4.1 Why Neo4j?

Neo4j is the leading graph database, offering:
- Native graph storage (not relational tables pretending to be graphs)
- Cypher query language (intuitive for graph patterns)
- Graph algorithms library (Leiden, PageRank, etc.)
- Python driver with good async support
- Vector index support (new in v5.11+) for hybrid retrieval

### 4.2 Docker Setup

```yaml
# Addition to docker-compose.yml
services:
  neo4j:
    image: neo4j:5.26.0  # Latest stable with vector support
    container_name: neo4j_rag
    ports:
      - "7474:7474"  # Browser UI
      - "7687:7687"  # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/password  # Change in production!
      NEO4J_PLUGINS: '["graph-data-science"]'  # For Leiden
      NEO4J_dbms_security_procedures_unrestricted: "gds.*,apoc.*"
    volumes:
      - ./neo4j_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### 4.3 Python Driver Connection

```python
from neo4j import GraphDatabase, AsyncGraphDatabase

class Neo4jClient:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def verify_connection(self) -> bool:
        """Verify Neo4j is accessible."""
        with self.driver.session() as session:
            result = session.run("RETURN 1")
            return result.single()[0] == 1

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

### 4.4 Schema Creation

```python
def create_schema(self):
    """Create indexes and constraints for GraphRAG."""
    constraints = [
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.book_id IS UNIQUE",
    ]

    indexes = [
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE INDEX chunk_book IF NOT EXISTS FOR (c:Chunk) ON (c.book_id)",
        # Vector index for hybrid retrieval
        """
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 3072,
            `vector.similarity_function`: 'cosine'
        }}
        """
    ]

    with self.driver.session() as session:
        for constraint in constraints:
            session.run(constraint)
        for index in indexes:
            session.run(index)
```

### 4.5 Entity and Relationship Creation

```python
def create_entity(
    self,
    name: str,
    entity_type: str,
    description: str,
    chunk_ids: List[str]
) -> None:
    """Create or merge an entity node."""
    query = """
    MERGE (e:Entity {name: $name})
    ON CREATE SET
        e.type = $type,
        e.description = $description,
        e.chunk_ids = $chunk_ids,
        e.mention_count = 1
    ON MATCH SET
        e.mention_count = e.mention_count + 1,
        e.chunk_ids = e.chunk_ids + $chunk_ids
    """
    with self.driver.session() as session:
        session.run(query, name=name, type=entity_type,
                   description=description, chunk_ids=chunk_ids)

def create_relationship(
    self,
    source: str,
    target: str,
    relation_type: str,
    description: str,
    strength: int,
    chunk_ids: List[str]
) -> None:
    """Create a relationship between entities."""
    query = """
    MATCH (s:Entity {name: $source})
    MATCH (t:Entity {name: $target})
    MERGE (s)-[r:RELATES_TO {type: $relation_type}]->(t)
    ON CREATE SET
        r.description = $description,
        r.strength = $strength,
        r.chunk_ids = $chunk_ids
    ON MATCH SET
        r.strength = CASE WHEN r.strength < $strength
                         THEN $strength ELSE r.strength END
    """
    with self.driver.session() as session:
        session.run(query, source=source, target=target,
                   relation_type=relation_type, description=description,
                   strength=strength, chunk_ids=chunk_ids)
```

### 4.6 Graph Queries for Retrieval

```python
def query_entity_neighborhood(
    self,
    entity_name: str,
    hops: int = 2,
    limit: int = 50
) -> List[str]:
    """Get chunk_ids from entity's neighborhood."""
    query = f"""
    MATCH (e:Entity {{name: $name}})-[*1..{hops}]-(related:Entity)
    WITH DISTINCT related
    UNWIND related.chunk_ids AS chunk_id
    RETURN DISTINCT chunk_id
    LIMIT $limit
    """
    with self.driver.session() as session:
        result = session.run(query, name=entity_name, limit=limit)
        return [record["chunk_id"] for record in result]

def query_related_entities(
    self,
    entity_name: str,
    relation_type: Optional[str] = None
) -> List[Dict]:
    """Find entities related to a given entity."""
    if relation_type:
        query = """
        MATCH (e:Entity {name: $name})-[r:RELATES_TO {type: $rel_type}]-(related)
        RETURN related.name AS name, related.type AS type,
               r.description AS relationship
        """
        params = {"name": entity_name, "rel_type": relation_type}
    else:
        query = """
        MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(related)
        RETURN related.name AS name, related.type AS type,
               r.type AS relation_type, r.description AS relationship
        ORDER BY r.strength DESC
        LIMIT 20
        """
        params = {"name": entity_name}

    with self.driver.session() as session:
        result = session.run(query, **params)
        return [dict(record) for record in result]
```

### 4.7 Community Detection with GDS

```python
def run_leiden_community_detection(self) -> Dict[str, int]:
    """Run Leiden algorithm and store community assignments."""
    # Create graph projection
    projection_query = """
    CALL gds.graph.project(
        'entityGraph',
        'Entity',
        {
            RELATES_TO: {
                type: 'RELATES_TO',
                properties: 'strength'
            }
        }
    )
    """

    # Run Leiden
    leiden_query = """
    CALL gds.leiden.write('entityGraph', {
        writeProperty: 'community',
        includeIntermediateCommunities: true,
        intermediateCommunitiesWriteProperty: 'communities_hierarchy',
        relationshipWeightProperty: 'strength'
    })
    YIELD communityCount, modularity
    RETURN communityCount, modularity
    """

    with self.driver.session() as session:
        session.run(projection_query)
        result = session.run(leiden_query)
        stats = result.single()

        # Clean up projection
        session.run("CALL gds.graph.drop('entityGraph')")

        return {
            "community_count": stats["communityCount"],
            "modularity": stats["modularity"]
        }
```

---

## 5. Integration with RAG1-Mini

### 5.1 Architecture Mapping

| GraphRAG Component | RAG1-Mini Equivalent | Integration Point |
|-------------------|---------------------|------------------|
| Text chunks | `section_chunker.py` output | Stage 4 |
| Entity extraction | New: `graph/extractor.py` | Stage 4.6 (new) |
| Graph storage | New: `graph/neo4j_client.py` | Stage 6b (new) |
| Community detection | Neo4j GDS | Stage 6b |
| Community summaries | New: `graph/summarizer.py` | Stage 6b |
| Graph query | New: `graph/query.py` | Stage 7 (retrieval) |
| Vector search | Existing: `weaviate_query.py` | Stage 7 |

### 5.2 File Structure

```
src/
+-- graph/                              # GraphRAG module
|   +-- __init__.py
|   +-- schemas.py                      # Pydantic models (GraphEntity, Community)
|   +-- extractor.py                    # LLM entity/relationship extraction
|   +-- auto_tuning.py                  # Domain-adaptive entity type discovery
|   +-- neo4j_client.py                 # Neo4j connection & MERGE operations
|   +-- community.py                    # Leiden detection + LLM summarization
|   +-- query.py                        # Hybrid graph retrieval
+-- stages/
|   +-- run_stage_4_5_autotune.py       # Auto-tuning extraction (recommended)
|   +-- run_stage_6b_neo4j.py           # Neo4j upload + Leiden + summaries
```

**Execution (see graphrag-quickstart.md):**
```bash
python -m src.stages.run_stage_4_5_autotune --strategy section  # Auto-discovers entity types
python -m src.stages.run_stage_6b_neo4j                         # Upload + Leiden + summarize
```

### 5.3 Data Flow

```
Existing Pipeline:
  Stage 1-3: PDF -> Clean Text -> Sentences
  Stage 4: Sentences -> Chunks
  Stage 5: Chunks -> Embeddings
  Stage 6: Embeddings -> Weaviate

GraphRAG Addition:
  Stage 4.6: Chunks -> Entities + Relationships (JSON)
  Stage 6b: Entities -> Neo4j Graph -> Communities -> Summaries

Query Time:
  1. Extract entities from query
  2. Query Neo4j for related chunks (graph traversal)
  3. Query Weaviate for similar chunks (vector search)
  4. Boost chunks found in both
  5. Generate answer with graph context
```

### 5.4 Hybrid Retrieval Strategy

```python
def query_graphrag(
    query: str,
    top_k: int = 10,
    graph_weight: float = 0.3,
    vector_weight: float = 0.7
) -> List[SearchResult]:
    """Combine graph traversal with vector search."""

    # 1. Extract entities from query using LLM
    query_entities = extract_entities_from_query(query)

    # 2. Get chunks via graph traversal
    graph_chunk_ids = set()
    for entity in query_entities:
        neighbors = neo4j_client.query_entity_neighborhood(
            entity.name, hops=2
        )
        graph_chunk_ids.update(neighbors)

    # 3. Get chunks via vector search
    vector_results = query_hybrid(
        query,
        top_k=top_k * 3,  # Oversample
        alpha=0.5
    )

    # 4. Score combination
    scored_results = {}
    for result in vector_results:
        base_score = result.score * vector_weight
        if result.chunk_id in graph_chunk_ids:
            base_score += graph_weight  # Graph boost
        scored_results[result.chunk_id] = (result, base_score)

    # 5. Sort and return top-k
    sorted_results = sorted(
        scored_results.values(),
        key=lambda x: x[1],
        reverse=True
    )
    return [r for r, s in sorted_results[:top_k]]
```

### 5.5 Global Query Mode

For questions like "What are the main themes across all books?":

```python
def global_query(query: str, level: int = 0) -> str:
    """Answer global questions using community summaries."""

    # 1. Get community summaries at specified level
    summaries = neo4j_client.get_community_summaries(level=level)

    # 2. If too many, use map-reduce
    if total_tokens(summaries) > 8000:
        return map_reduce_answer(query, summaries)

    # 3. Otherwise, direct synthesis
    context = "\n\n".join([
        f"## {s['title']}\n{s['summary']}"
        for s in summaries
    ])

    return generate_answer(query, context)
```

---

## 6. Comparison: GraphRAG vs RAPTOR

| Aspect | RAPTOR | GraphRAG |
|--------|--------|----------|
| **Structure** | Document tree (per-book) | Knowledge graph (cross-book) |
| **Focus** | Hierarchical document summaries | Entity relationships |
| **Best for** | Single-document deep questions | Cross-document synthesis |
| **Index time** | Fast (clustering + summarization) | Slower (entity extraction) |
| **Query time** | Fast (collapsed tree search) | Medium (graph + vector) |
| **Storage** | Weaviate only | Weaviate + Neo4j |
| **Complexity** | Medium | High |

**Recommendation:** Implement RAPTOR first (simpler), then add GraphRAG for cross-document capabilities. They complement each other well.

---

## 7. References

### Primary Sources
- [GraphRAG Paper (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130)
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

### Neo4j Resources
- [Neo4j GraphRAG Python Package](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- [Neo4j Python Driver Manual](https://neo4j.com/docs/python-manual/current/)
- [Leiden Algorithm in GDS](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)

### Related Techniques
- [Leiden Algorithm Wikipedia](https://en.wikipedia.org/wiki/Leiden_algorithm)
- [RAPTOR (arXiv:2401.18059)](https://arxiv.org/abs/2401.18059) - Complementary approach
- [GraphRAG Auto-Tuning](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/)

### Implementation Examples
- [Analytics Vidhya Neo4j GraphRAG Tutorial](https://www.analyticsvidhya.com/blog/2024/11/graphrag-with-neo4j/)
- [Weaviate GraphRAG Blog](https://weaviate.io/blog/graph-rag)
- [Qdrant + Neo4j GraphRAG Example](https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/)

---

## 8. Implementation Analysis (Dec 2025)

### 8.1 Research Sources

This analysis compares RAG1-Mini's GraphRAG implementation against:
- Original paper: [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
- Survey paper: [arXiv:2501.00309](https://arxiv.org/abs/2501.00309)
- Microsoft implementation: [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)
- Microsoft documentation: [microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/)
- Entity resolution discussion: [GitHub Issue #847](https://github.com/microsoft/graphrag/issues/847)

### 8.2 Critical Deviations from Original Paper

#### Entity Resolution (HIGH PRIORITY)

**Original GraphRAG:**
- Uses "exact string matching" but notes "softer matching approaches can be used"
- Microsoft had entity resolution step but removed it due to inconsistency
- Paper states GraphRAG is "generally resilient to duplicate entities since duplicates are typically clustered together for summarization"

**This Project (`src/graph/schemas.py:66-68`):**
```python
def normalized_name(self) -> str:
    return self.name.strip().lower()
```

**Problems:**
- "Marcus Aurelius" and "Aurelius" become different entities
- "dopamine system", "the dopamine", "DOPAMINE" all separate
- No Unicode normalization (NFKC)
- No stopword removal
- No lemmatization

**Recommended Fix:**
- Unicode NFKC normalization
- Remove leading/trailing stopwords (the, a, an, of, etc.)
- Strip punctuation
- Normalize whitespace

#### No Map-Reduce for Global Search (HIGH)

**Original GraphRAG (from paper):**
1. Shuffle community summaries into pre-specified token chunks
2. Generate partial answers in parallel with 0-100 helpfulness scores
3. Filter zero-score answers
4. Sort by helpfulness, combine top answers into final response

**This Project (`src/graph/query.py:284-296`):**
```python
# Score communities by keyword overlap with query
query_words = set(query.lower().split())
for community in communities:
    summary_words = set(community.summary.lower().split())
    overlap = len(query_words & summary_words)
```

- Simple keyword matching (not map-reduce)
- No partial answer generation
- No helpfulness scoring
- No embedding similarity on summaries

#### Community Hierarchy Not Utilized (MEDIUM)

**Original GraphRAG:**
- Multi-level hierarchy (C0, C1, C2...) from Leiden
- Bottom-up summarization: leaf communities first, then aggregate
- Query-time: Select community level based on query scope

**This Project (`src/graph/community.py:443-446`):**
```python
community = Community(
    community_id=community_key,
    level=0,  # Single level for now  <-- HARDCODED
    ...
)
```

- Stores `includeIntermediateCommunities=True` but discards hierarchy
- All communities stored at level=0
- No level selection at query time

#### No Self-Reflection/Glean Loop (MEDIUM)

**Original GraphRAG:**
- LLM evaluates: "Are there any entities or relationships you missed?"
- Uses logit bias to force yes/no response
- Iterates up to N times (~3 iterations)

**This Project:**
- Single-pass extraction (`src/graph/extractor.py:49-100`)
- May miss 20-30% of entities that require re-reading

### 8.3 Missing Features

| Feature | Original | This Project | Impact |
|---------|----------|--------------|--------|
| Claims/Covariates extraction | Verifiable facts per entity | Not implemented | Reduced factual grounding |
| Self-reflection loop | 3 iterations per chunk | Single pass | ~20% fewer entities |
| Community hierarchy | Multi-level (C0-Cn) | Level 0 only | Less query flexibility |
| Map-reduce global search | Parallel partial answers | Keyword matching | Worse global query answers |
| Embedding on summaries | Vector similarity | Word overlap | Semantic matching lost |
| Dynamic relationship types | Neo4j edge labels | Single RELATED_TO | Schema less explicit |

### 8.4 Correct Implementations

| Component | Status | Notes |
|-----------|--------|-------|
| Leiden algorithm | Correct | Uses Neo4j GDS with gamma=1.0, maxLevels=10 |
| GDS graph projection | Correct | Undirected projection for community detection |
| Neo4j MERGE pattern | Correct | Idempotent entity/relationship upserts |
| Structured LLM output | Correct | Pydantic + JSON Schema enforcement |
| Query entity extraction | Good | LLM + regex fallback + Neo4j validation |
| Auto-tuning entity types | Good | Stratified consolidation for multi-corpus |
| Incremental saving | Correct | Resume support after crash |

### 8.5 Minor Issues

**Issue 1:** Relationship weight not used in Leiden
```python
# src/graph/community.py:77-85
graph, result = gds.graph.project(
    graph_name,
    "Entity",
    {"RELATED_TO": {"orientation": "UNDIRECTED"}},  # No weight projection
)
```
Fix: Add `properties: 'weight'` and `relationshipWeightProperty` to Leiden call.

**Issue 2:** Description merge strategy (longer wins)
```python
# src/graph/neo4j_client.py - ON MATCH
e.description = CASE
    WHEN size(entity.description) > size(coalesce(e.description, ''))
    THEN entity.description
    ELSE e.description
END
```
Problem: Longer isn't always better. Could lose important context.

### 8.6 Configuration Comparison

| Parameter | Original Paper | This Project | Status |
|-----------|---------------|--------------|--------|
| Chunk size | 600 tokens | 800 tokens | OK (section-aware) |
| Chunk overlap | 100 tokens | 2 sentences | OK |
| Context window | 8,000 tokens | 6,000 chars (~1500 tokens) | Should increase |
| Leiden resolution | 1.0 | 1.0 | Correct |
| Max hierarchy levels | 10 | 10 | Correct |
| Min community size | Not specified | 3 | OK |

### 8.7 Recommended Improvements (Priority Order)

1. **P1: Enhanced Entity Resolution** - Add Unicode normalization, stopword removal, punctuation stripping to `normalized_name()`
2. **P2: Map-Reduce Global Search** - Implement proper global search with parallel partial answers and helpfulness scoring
3. **P3: Community Hierarchy** - Store and use multi-level community structure from Leiden
4. **P4: Embedding Similarity for Communities** - Replace keyword matching with embedding similarity on summaries
5. **P5: Self-Reflection Loop** - Add iterative extraction with "did I miss anything?" evaluation

### 8.8 What Works Well

- Core Leiden pipeline is correctly implemented
- Neo4j integration is solid with proper MERGE patterns
- Auto-tuning with stratified consolidation is a good addition not in original paper
- Query entity extraction with LLM + fallback chain is robust
- Crash recovery with --resume is well implemented

---

*Last Updated: 2025-12-28*
