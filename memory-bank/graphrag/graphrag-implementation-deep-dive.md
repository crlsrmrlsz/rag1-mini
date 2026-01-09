# GraphRAG: Complete Implementation Deep-Dive

**Date:** 2026-01-09
**Status:** Implementation Analysis Complete + Paper Alignment Update
**Related:** [graphrag.md](graphrag.md), [graphrag-tutorial.md](graphrag-tutorial.md), [fix_graphrag.md](fix_graphrag.md), [plan_correct_1.md](plan_correct_1.md)

---

## Executive Summary

GraphRAG is a hybrid retrieval strategy that combines **knowledge graph traversal** with **vector search** and uses **Reciprocal Rank Fusion (RRF)** to merge results. This document provides a complete code-level analysis of the implementation.

**Latest Update (2026-01-09):** Paper alignment features added:
- **Hierarchical Communities**: C0, C1, C2 levels parsed from Leiden's `intermediateCommunityIds`
- **PageRank Centrality**: Computed via Neo4j GDS, stored on Entity nodes
- **Structured Relationships**: `CommunityRelationship` objects in JSON storage
- **Map-Reduce for Global Queries**: Async parallel map + reduce synthesis

---

## 1. Two-Phase Architecture

```
PHASE 1: INDEXING (Offline)
═══════════════════════════════════════════════════════════════════════════════

   Stage 4                Stage 4.5                   Stage 6b
   ┌─────────┐           ┌─────────────────┐         ┌───────────────────────────┐
   │ Chunks  │──────────>│ Entity          │────────>│ Neo4j Upload + Leiden     │
   │ (JSON)  │           │ Extraction      │         │ + Community Summaries     │
   └─────────┘           │ (LLM per chunk) │         │ + Weaviate Embeddings     │
                         └─────────────────┘         └───────────────────────────┘

   Output Files:
   • data/processed/07_graph/extraction_results.json  (entities + relationships)
   • data/processed/07_graph/discovered_types.json    (consolidated taxonomy)
   • data/processed/07_graph/communities.json         (summaries backup)
   • data/processed/07_graph/leiden_checkpoint.json   (crash recovery)
   • Weaviate: Community_section800_v1 collection     (embeddings)
   • Neo4j: Entity nodes + RELATED_TO edges           (graph)

PHASE 2: QUERY (Online)
═══════════════════════════════════════════════════════════════════════════════

   User Query
       │
       ├──────────────────┬──────────────────┬────────────────────┐
       ▼                  ▼                  ▼                    ▼
   ┌──────────┐      ┌──────────────┐   ┌──────────┐       ┌────────────┐
   │ LLM      │      │ Neo4j        │   │ Weaviate │       │ Community  │
   │ Entity   │─────>│ Graph        │   │ Vector   │       │ Retrieval  │
   │ Extract  │      │ Traversal    │   │ Search   │       │            │
   └──────────┘      └──────────────┘   └──────────┘       └────────────┘
                            │                  │                   │
                            └─────────┬────────┘                   │
                                      ▼                            │
                              ┌───────────────┐                    │
                              │ RRF Merge     │                    │
                              │ (boost overlaps)                   │
                              └───────────────┘                    │
                                      │                            │
                                      └──────────────┬─────────────┘
                                                     ▼
                                            ┌───────────────┐
                                            │ Answer Gen    │
                                            │ (chunks +     │
                                            │  communities) │
                                            └───────────────┘
```

---

## 2. Indexing Phase: Code Analysis

### Stage 4.5: Auto-Tuning Entity Extraction

**File:** `src/graph/auto_tuning.py`

```python
# Step 1: Per-chunk extraction with open-ended types
def extract_chunk(chunk: dict, model: str) -> OpenExtractionResult:
    """LLM extracts entities with freely-assigned types."""
    prompt = GRAPHRAG_OPEN_EXTRACTION_PROMPT.format(
        text=chunk["text"],
        max_entities=50,
        max_relationships=50,
    )
    return call_structured_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        response_model=OpenExtractionResult,  # Pydantic enforces JSON schema
        temperature=0.0,
    )
```

**Consolidation Strategies:**

| Strategy | Algorithm | Use Case |
|----------|-----------|----------|
| `global` | Rank by total count | Single-domain corpora |
| `stratified` | Top-K from each corpus | Mixed corpora (recommended) |

**Data Flow:**
```
Chunks ─> Per-book extraction ─> merge_extractions() ─> extraction_results.json
                                                     ─> consolidate_stratified() ─> discovered_types.json
```

### Stage 6b: Neo4j + Leiden + Summarization

**File:** `src/graph/neo4j_client.py`

#### Entity Upload with Normalization

```python
def upload_entities(driver: Driver, entities: list[dict], batch_size: int = 100):
    # CRITICAL: Pre-compute normalized names in Python
    for entity in entities:
        ge = GraphEntity(name=entity["name"], entity_type=entity.get("entity_type", ""))
        entity["normalized_name"] = ge.normalized_name()

    query = """
    UNWIND $entities AS entity
    MERGE (e:Entity {normalized_name: entity.normalized_name})
    ON CREATE SET e.name = entity.name, e.entity_type = entity.entity_type, ...
    """
```

**Normalization Logic (schemas.py:72-101):**

```python
def normalized_name(self) -> str:
    name = unicodedata.normalize('NFKC', self.name.strip())  # café → cafe
    name = name.lower()

    # Remove leading/trailing stopwords
    words = name.split()
    while words and words[0] in EDGE_STOPWORDS:  # {'the', 'a', 'an', ...}
        words.pop(0)
    while words and words[-1] in EDGE_STOPWORDS:
        words.pop()

    name = re.sub(r'[^\w\s]', '', ' '.join(words))  # Remove punctuation
    return ' '.join(name.split())
```

#### Leiden Community Detection

**File:** `src/graph/community.py`

```python
def run_leiden(gds, graph, resolution=1.0, seed=42, concurrency=1):
    """DETERMINISTIC settings for reproducibility."""
    result = gds.leiden.stream(
        graph,
        gamma=resolution,
        randomSeed=seed,         # FIXED SEED
        concurrency=concurrency, # SINGLE THREAD
    )
    # Same graph + same seed = IDENTICAL community assignments
```

#### Community Summarization

```python
def summarize_community(members, relationships, model):
    context = build_community_context(members, relationships)
    summary = call_chat_completion(...)
    embedding = embed_texts([summary])[0]  # For vector retrieval
    return summary, embedding
```

---

## 3. Query Phase: Hybrid Retrieval with RRF

### Complete Query Flow

```
Query: "How does dopamine affect motivation?"
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: ENTITY EXTRACTION (query.py:85-127)                                 │
│                                                                             │
│   extract_query_entities_llm(query) → ["dopamine", "motivation"]            │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: NEO4J ENTITY LOOKUP (neo4j_client.py:445-482)                       │
│                                                                             │
│   find_entities_by_names(driver, ["dopamine", "motivation"])                │
│   # Pre-normalize using SAME logic as upload                                │
│   → Matched: ["dopamine"]                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: GRAPH TRAVERSAL (neo4j_client.py:396-442)                           │
│                                                                             │
│   find_entity_neighbors(driver, "dopamine", max_hops=2)                     │
│   → graph_chunk_ids = ["behave::chunk_42", "behave::chunk_43", ...]        │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ├───────────────────────────────────────────┐
        ▼                                           ▼
┌───────────────────────────┐     ┌───────────────────────────────────────────┐
│ STEP 4a: VECTOR SEARCH    │     │ STEP 4b: FETCH GRAPH CHUNKS               │
│ (Weaviate)                │     │ (query.py:264-330)                        │
│                           │     │                                           │
│ Top-k by embedding sim    │     │ fetch_chunks_by_ids(graph_chunk_ids)      │
│                           │     │ → all_graph_chunks                        │
└───────────────────────────┘     └───────────────────────────────────────────┘
        │                                           │
        │                                           ▼
        │                         ┌───────────────────────────────────────────┐
        │                         │ STEP 4c: RANK BY PATH_LENGTH              │
        │                         │ (query.py:527-567)                        │
        │                         │                                           │
        │                         │ _build_graph_ranked_list(graph_context,   │
        │                         │                          fetched_chunks)  │
        │                         │                                           │
        │                         │ Sort by path_length (shorter = higher)    │
        │                         └───────────────────────────────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: RRF MERGE (rrf.py:60-149)                                           │
│                                                                             │
│   reciprocal_rank_fusion(                                                   │
│       result_lists=[vector_search_results, graph_ranked_results],           │
│       query_types=["vector", "graph"],                                      │
│       k=60, top_k=10                                                        │
│   )                                                                         │
│                                                                             │
│   RRF Formula: score(chunk) = Σ 1/(k + rank + 1)                            │
│                                                                             │
│   Chunks in BOTH lists get summed scores → BOOSTED                          │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: COMMUNITY CONTEXT (query.py:351-458)                                │
│                                                                             │
│   retrieve_community_context(query, top_k=3)                                │
│   → Top communities by embedding similarity                                 │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: ANSWER GENERATION                                                   │
│                                                                             │
│   graph_context = format_graph_context_for_generation(metadata)             │
│   → LLM prompt includes: community summaries + entity relationships         │
│                          + RRF-merged chunks + question                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. RRF Deep-Dive

### The RRF Algorithm (rrf.py)

```python
def reciprocal_rank_fusion(result_lists, query_types, k=60, top_k=10):
    scores = defaultdict(float)

    for query_idx, results in enumerate(result_lists):
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.chunk_id] += rrf_score

    # Chunks in BOTH lists get summed scores → BOOSTED
```

### Visual Example

```
VECTOR RESULTS              GRAPH RESULTS (by path_length)
Rank 0: chunk_A             Rank 0: chunk_X
Rank 1: chunk_B             Rank 1: chunk_A ← SAME!
Rank 2: chunk_C             Rank 2: chunk_Y
Rank 3: chunk_D             Rank 3: chunk_B ← SAME!

RRF CALCULATION (k=60):
chunk_A: 1/(60+0+1) + 1/(60+1+1) = 0.0164 + 0.0161 = 0.0325 ★ HIGHEST
chunk_B: 1/(60+1+1) + 1/(60+3+1) = 0.0161 + 0.0156 = 0.0317 ★ SECOND
chunk_X: 0          + 1/(60+0+1) = 0.0164 (graph only)
chunk_C: 1/(60+2+1) + 0          = 0.0159 (vector only)

KEY INSIGHT: Chunks in BOTH vector AND graph get boosted, capturing
both semantic relevance AND structural importance in knowledge graph.
```

---

## 5. Critical Fixes Applied

| # | Issue | Fix | Commit |
|---|-------|-----|--------|
| 1 | Docstrings claimed RRF, implementation used simple boost | Proper RRF with `reciprocal_rank_fusion()` | 6e8a353 |
| 2 | Graph-only chunks never fetched | `fetch_chunks_by_ids(graph_chunk_ids)` | 6e8a353 |
| 3 | Entity normalization mismatch between upload/query | Pre-normalize in Python before Cypher query | 6e8a353 |
| 6 | Community context not in generation prompt | Added `graph_context` parameter to `generate_answer()` | 81a5ce9 |

---

## 6. Configuration Reference

```python
# Entity Extraction
GRAPHRAG_EXTRACTION_MODEL = "openrouter/anthropic/claude-3-5-haiku"
GRAPHRAG_MAX_ENTITIES = 50
GRAPHRAG_MAX_RELATIONSHIPS = 50

# Auto-Tuning
GRAPHRAG_TYPES_PER_CORPUS = 15
GRAPHRAG_MIN_CORPUS_PERCENTAGE = 0.1

# Leiden Algorithm
GRAPHRAG_LEIDEN_RESOLUTION = 1.0
GRAPHRAG_LEIDEN_SEED = 42
GRAPHRAG_LEIDEN_CONCURRENCY = 1
GRAPHRAG_MIN_COMMUNITY_SIZE = 3

# Hierarchy (NEW)
GRAPHRAG_MAX_HIERARCHY_LEVELS = 3  # C0, C1, C2
GRAPHRAG_LEVEL_MIN_SIZES = {0: 3, 1: 5, 2: 10}

# PageRank (NEW)
GRAPHRAG_PAGERANK_DAMPING = 0.85
GRAPHRAG_PAGERANK_ITERATIONS = 20

# Map-Reduce (NEW)
GRAPHRAG_MAP_REDUCE_TOP_K = 5  # Communities for map phase
GRAPHRAG_MAP_MAX_TOKENS = 300  # Per-community partial answer
GRAPHRAG_REDUCE_MAX_TOKENS = 500  # Final synthesis

# Query Time
GRAPHRAG_TRAVERSE_DEPTH = 2
GRAPHRAG_TOP_COMMUNITIES = 3
GRAPHRAG_RRF_K = 60
```

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `src/graph/schemas.py` | Pydantic models, `GraphEntity.normalized_name()`, `CommunityRelationship` |
| `src/graph/auto_tuning.py` | Entity extraction, `run_auto_tuning()` |
| `src/graph/neo4j_client.py` | Graph DB ops, `upload_entities()`, `find_entity_neighbors()` |
| `src/graph/community.py` | Leiden + summaries, `run_leiden()`, `summarize_community()` |
| `src/graph/hierarchy.py` | Multi-level parsing, `parse_leiden_hierarchy()`, `CommunityLevel` |
| `src/graph/centrality.py` | PageRank, `compute_pagerank()`, `write_pagerank_to_neo4j()` |
| `src/graph/map_reduce.py` | Global queries, `map_reduce_global_query()`, `classify_query()` |
| `src/graph/query.py` | Hybrid retrieval, `hybrid_graph_retrieval()`, `hybrid_graph_retrieval_with_map_reduce()` |
| `src/rag_pipeline/retrieval/rrf.py` | RRF algorithm, `reciprocal_rank_fusion()` |

---

## 8. Paper Alignment Features (NEW)

### 8.1 Hierarchical Communities

**File:** `src/graph/hierarchy.py`

The Leiden algorithm produces multi-level community assignments via `intermediateCommunityIds`. This module parses them into a structured hierarchy:

```python
@dataclass
class CommunityLevel:
    level: int  # 0=finest, 1=medium, 2=coarsest
    communities: dict[int, set[int]]  # community_id -> node_ids
    parent_map: dict[int, int]  # child_community -> parent_community
    child_map: dict[int, list[int]]  # parent -> children

def parse_leiden_hierarchy(leiden_result: dict, max_levels: int = 3) -> dict[int, CommunityLevel]:
    """Extract C0, C1, C2 from Leiden's intermediate_ids."""
    # intermediate_ids[0] = coarsest, reversed to get fine-to-coarse
```

### 8.2 PageRank Centrality

**File:** `src/graph/centrality.py`

PageRank identifies "hub" entities that connect to other important entities:

```python
def compute_pagerank(gds: GraphDataScience, graph: Any) -> dict[int, float]:
    """Run PageRank via Neo4j GDS."""
    result = gds.pageRank.stream(graph, dampingFactor=0.85, maxIterations=20)
    return {record.nodeId: record.score for record in result.itertuples()}

def write_pagerank_to_neo4j(driver: Driver, pagerank_scores: dict[int, float]) -> int:
    """Store PageRank on Entity nodes for query-time access."""
```

### 8.3 Structured Relationships

**File:** `src/graph/schemas.py`

Relationships are now stored as structured objects, not just counts:

```python
class CommunityRelationship(BaseModel):
    source: str  # Source entity name
    target: str  # Target entity name
    relationship_type: str  # e.g., "CAUSES", "MODULATES"
    description: str  # Optional description
    weight: float  # 0.0-1.0

class Community(BaseModel):
    # ... existing fields ...
    relationships: list[CommunityRelationship]  # NEW: structured relationships
    parent_id: Optional[str]  # NEW: parent at coarser level
```

### 8.4 Map-Reduce for Global Queries

**File:** `src/graph/map_reduce.py`

Global queries (thematic, corpus-wide) use map-reduce over community summaries:

```python
def classify_query(query: str) -> str:
    """LLM-based classification: 'local' or 'global'."""

async def map_reduce_global_query_async(query: str, communities: list[Community]) -> MapReduceResult:
    """
    Map Phase: Parallel partial answers from each community (async).
    Reduce Phase: Synthesize partial answers into final response.
    """
    # MAP: async parallel LLM calls
    tasks = [_map_community_async(query, community) for community in communities]
    map_results = await asyncio.gather(*tasks)

    # REDUCE: single LLM call to synthesize
    final_answer = call_chat_completion(reduce_prompt)

    return MapReduceResult(
        final_answer=final_answer,
        partial_answers=partial_answers,
        communities_used=communities_used,
        map_time_ms=map_time_ms,
        reduce_time_ms=reduce_time_ms,
    )
```

### 8.5 Enhanced Query Flow

**File:** `src/graph/query.py`

The new `hybrid_graph_retrieval_with_map_reduce()` function extends hybrid retrieval:

```python
def hybrid_graph_retrieval_with_map_reduce(query: str, driver: Driver, vector_results: list[dict], ...) -> tuple[list[dict], dict]:
    """
    For local queries: Standard RRF merge (entity traversal + vector)
    For global queries: Map-reduce over community summaries
    """
    # Determine if global query
    if use_map_reduce and should_use_map_reduce(query, extracted_entities):
        communities = retrieve_communities_for_map_reduce(query, top_k=5)
        mr_result = map_reduce_global_query(query, communities)
        metadata["map_reduce_result"] = {...}  # Includes timing, partial answers
        return vector_results, metadata

    # Fall back to standard hybrid retrieval
    return hybrid_graph_retrieval(query, driver, vector_results)
```

---

*Last Updated: 2026-01-09*
