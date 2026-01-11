# GraphRAG: Complete Deep Dive

[← Query Decomposition](query-decomposition.md) | [Home](../../README.md)

> **Paper:** [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) | Edge et al. (Microsoft Research) | April 2024

GraphRAG augments traditional RAG with knowledge graph structures to enable cross-document synthesis and thematic understanding. This document explains every detail of the implementation.

---

## Table of Contents

1. [The Problem GraphRAG Solves](#the-problem-graphrag-solves)
2. [High-Level Architecture](#high-level-architecture)
3. [Phase 1: Indexing Pipeline (Offline)](#phase-1-indexing-pipeline-offline)
   - [Stage 4.5: Entity Extraction](#stage-45-entity-extraction)
   - [Stage 6b: Graph Construction & Community Detection](#stage-6b-graph-construction--community-detection)
4. [Phase 2: Query Pipeline (Online)](#phase-2-query-pipeline-online)
   - [Entity Extraction from Query](#step-1-entity-extraction-from-query)
   - [Graph Traversal](#step-2-graph-traversal)
   - [RRF Merging](#step-3-rrf-merging)
   - [Map-Reduce for Global Queries](#step-4-map-reduce-for-global-queries)
5. [Key Algorithms Explained](#key-algorithms-explained)
6. [Configuration Reference](#configuration-reference)
7. [Running the Pipeline](#running-the-pipeline)
8. [Performance & Cost Analysis](#performance--cost-analysis)

---

## The Problem GraphRAG Solves

Vector search finds chunks with similar embeddings to the query. This works well for **local queries**:

```
Query: "What is dopamine?"
       ↓ embedding similarity
Result: Chunks mentioning dopamine directly
```

But vector search **fails on global queries** that require synthesis across documents:

```
Query: "What are the main themes across all 19 books?"
       ↓ embedding similarity
Result: ??? (No single chunk contains this answer)
```

**GraphRAG's solution:**
1. Build a knowledge graph of entities and relationships
2. Detect communities of related entities using Leiden algorithm
3. Generate summaries for each community
4. For global queries: Map-reduce over community summaries

---

## High-Level Architecture

```
                         GRAPHRAG COMPLETE ARCHITECTURE
===============================================================================

PHASE 1: INDEXING (Offline - ~10 hours for 19 books)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Stage 4                 Stage 4.5                      Stage 6b
  ┌─────────────┐        ┌─────────────────────┐        ┌─────────────────────────┐
  │   Chunks    │───────▶│  Entity Extraction  │───────▶│  Neo4j Upload           │
  │   (JSON)    │        │  (LLM per chunk)    │        │  + Leiden Algorithm     │
  │             │        │                     │        │  + PageRank Centrality  │
  │  ~5000      │        │  Discovers entity   │        │  + Community Summaries  │
  │  chunks     │        │  types from corpus  │        │  + Entity Embeddings    │
  └─────────────┘        └─────────────────────┘        └─────────────────────────┘
                                   │                              │
                                   ▼                              ▼
                         ┌─────────────────────┐        ┌─────────────────────────┐
                         │ extraction_results  │        │  Neo4j Graph Database   │
                         │ .json               │        │  ┌─────┐    ┌─────┐     │
                         │                     │        │  │Ent1 │───▶│Ent2 │     │
                         │ discovered_types    │        │  └─────┘    └─────┘     │
                         │ .json               │        │     │          │        │
                         └─────────────────────┘        │     ▼          ▼        │
                                                        │  ┌─────┐    ┌─────┐     │
                                                        │  │Ent3 │───▶│Ent4 │     │
                                                        │  └─────┘    └─────┘     │
                                                        └─────────────────────────┘
                                                                   │
                                                                   ▼
                                                        ┌─────────────────────────┐
                                                        │  Weaviate Collections   │
                                                        │                         │
                                                        │  Entity_section800_v1   │
                                                        │  (entity embeddings)    │
                                                        │                         │
                                                        │  Community_section800_v1│
                                                        │  (summary embeddings)   │
                                                        └─────────────────────────┘


PHASE 2: QUERY (Online - ~1-2 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────────────────────────────┐
  │                            USER QUERY                                     │
  │                   "How does dopamine affect motivation?"                  │
  └──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     STEP 1: ENTITY EXTRACTION                            │
  │                                                                          │
  │   Primary: Embedding similarity search (~50ms)                           │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │  Query: "How does dopamine affect motivation?"                  │   │
  │   │         ↓ embed query                                           │   │
  │   │  [0.23, 0.45, 0.12, ...]                                        │   │
  │   │         ↓ cosine similarity search in Weaviate                  │   │
  │   │  Entities: ["dopamine", "motivation", "reward system"]          │   │
  │   └─────────────────────────────────────────────────────────────────┘   │
  │                                                                          │
  │   Fallback: LLM extraction (if embedding returns empty, ~1-2s)           │
  │   ┌─────────────────────────────────────────────────────────────────┐   │
  │   │  LLM: "Extract entity mentions from: 'How does dopamine...'"   │   │
  │   │  Response: {"entities": ["dopamine", "motivation"]}             │   │
  │   └─────────────────────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     STEP 2: PARALLEL RETRIEVAL                           │
  │                                                                          │
  │   ┌────────────────────┐ ┌────────────────────┐ ┌──────────────────────┐ │
  │   │  WEAVIATE VECTOR   │ │   NEO4J GRAPH      │ │  WEAVIATE COMMUNITY  │ │
  │   │  SEARCH            │ │   TRAVERSAL        │ │  RETRIEVAL           │ │
  │   │                    │ │                    │ │                      │ │
  │   │  Query → Embed →   │ │  "dopamine" →      │ │  Query → Embed →     │ │
  │   │  HNSW search →     │ │  2-hop neighbors → │ │  Top-3 community     │ │
  │   │  Top-20 chunks     │ │  Related chunks    │ │  summaries           │ │
  │   └────────────────────┘ └────────────────────┘ └──────────────────────┘ │
  │           │                       │                        │             │
  │           ▼                       ▼                        ▼             │
  │   ┌────────────────┐     ┌────────────────┐      ┌────────────────────┐  │
  │   │ chunk_1 (0.92) │     │ chunk_42       │      │ "Dopamine and the  │  │
  │   │ chunk_7 (0.89) │     │ chunk_43       │      │  reward system..." │  │
  │   │ chunk_12 (0.87)│     │ chunk_100      │      └────────────────────┘  │
  │   │ ...            │     │ ...            │                              │
  │   └────────────────┘     └────────────────┘                              │
  └──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     STEP 3: RRF MERGE                                    │
  │                                                                          │
  │   Vector Results (ranked by similarity):                                 │
  │   ┌────────────────────────────────────────────────────────────────┐    │
  │   │  Rank 1: chunk_1  (score: 0.92)                                │    │
  │   │  Rank 2: chunk_7  (score: 0.89)                                │    │
  │   │  Rank 3: chunk_12 (score: 0.87)   ◀── Also in graph results!  │    │
  │   │  Rank 4: chunk_23 (score: 0.85)                                │    │
  │   │  Rank 5: chunk_42 (score: 0.83)   ◀── Also in graph results!  │    │
  │   └────────────────────────────────────────────────────────────────┘    │
  │                                                                          │
  │   Graph Results (ranked by path_length):                                 │
  │   ┌────────────────────────────────────────────────────────────────┐    │
  │   │  Rank 1: chunk_42 (path_length: 1)  ◀── Direct neighbor       │    │
  │   │  Rank 2: chunk_12 (path_length: 1)  ◀── Direct neighbor       │    │
  │   │  Rank 3: chunk_100 (path_length: 2) ◀── 2-hop neighbor        │    │
  │   │  Rank 4: chunk_150 (path_length: 2)                            │    │
  │   └────────────────────────────────────────────────────────────────┘    │
  │                                                                          │
  │   RRF Formula: score = 1/(k + rank_vector) + 1/(k + rank_graph)         │
  │   ┌────────────────────────────────────────────────────────────────┐    │
  │   │  chunk_42: 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318     │    │
  │   │  chunk_12: 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320 ◀──│    │
  │   │  chunk_1:  1/(60+1) + 0        = 0.0164 + 0      = 0.0164 ★  │    │
  │   │  chunk_100: 0       + 1/(60+3) = 0      + 0.0159 = 0.0159     │    │
  │   │                                                                │    │
  │   │  ★ Chunks in BOTH lists get BOOSTED (higher combined score)   │    │
  │   └────────────────────────────────────────────────────────────────┘    │
  │                                                                          │
  │   Final Merged Results:                                                  │
  │   1. chunk_12 (0.0320) - In both lists, boosted                         │
  │   2. chunk_42 (0.0318) - In both lists, boosted                         │
  │   3. chunk_1  (0.0164) - Vector only                                    │
  │   4. chunk_100 (0.0159) - Graph only                                    │
  └──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     STEP 4: ANSWER GENERATION                            │
  │                                                                          │
  │   Context = Community Summaries + Entity Relationships + Merged Chunks   │
  │                                                                          │
  │   ┌────────────────────────────────────────────────────────────────┐    │
  │   │ ## Background (from knowledge graph)                           │    │
  │   │                                                                │    │
  │   │ Relevant Themes:                                               │    │
  │   │ - Dopamine is central to the brain's reward circuitry...      │    │
  │   │                                                                │    │
  │   │ Related Concepts:                                              │    │
  │   │ - dopamine → REGULATES → motivation                           │    │
  │   │ - reward system → INVOLVES → dopamine                         │    │
  │   │                                                                │    │
  │   │ ## Retrieved Passages                                          │    │
  │   │ [chunk_12]: "Dopamine neurons fire in anticipation of..."     │    │
  │   │ [chunk_42]: "The mesolimbic pathway connects..."              │    │
  │   │ ...                                                            │    │
  │   │                                                                │    │
  │   │ ## Question                                                    │    │
  │   │ How does dopamine affect motivation?                           │    │
  │   └────────────────────────────────────────────────────────────────┘    │
  │                                                                          │
  │                              ↓ LLM                                       │
  │                                                                          │
  │   ┌────────────────────────────────────────────────────────────────┐    │
  │   │ Dopamine affects motivation through the mesolimbic pathway...  │    │
  │   └────────────────────────────────────────────────────────────────┘    │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Indexing Pipeline (Offline)

### Stage 4.5: Entity Extraction

**Purpose:** Extract entities and relationships from each chunk using LLM.

```
RUN: python -m src.stages.run_stage_4_5_autotune --strategy section
```

#### Detailed Flow

```
                    ENTITY EXTRACTION FLOW
════════════════════════════════════════════════════════════════

  INPUT: 5000 chunks from Stage 4 (section chunking)

  ┌─────────────────────────────────────────────────────────────┐
  │                    PER-CHUNK EXTRACTION                     │
  │                                                             │
  │  For each chunk:                                            │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  Chunk Text:                                          │  │
  │  │  "Sapolsky explains that dopamine neurons fire not   │  │
  │  │   when receiving a reward, but in anticipation of    │  │
  │  │   the reward. This creates the wanting-liking gap."  │  │
  │  └───────────────────────────────────────────────────────┘  │
  │                           │                                 │
  │                           ▼                                 │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  LLM Prompt (structured output):                      │  │
  │  │                                                       │  │
  │  │  "Extract entities and relationships from:            │  │
  │  │   {chunk_text}                                        │  │
  │  │                                                       │  │
  │  │   Return JSON with:                                   │  │
  │  │   - entities: [{name, entity_type, description}]      │  │
  │  │   - relationships: [{source, target, type, desc}]"    │  │
  │  └───────────────────────────────────────────────────────┘  │
  │                           │                                 │
  │                           ▼                                 │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  LLM Response (OpenExtractionResult):                 │  │
  │  │                                                       │  │
  │  │  entities: [                                          │  │
  │  │    {name: "Sapolsky",                                 │  │
  │  │     entity_type: "RESEARCHER",                        │  │
  │  │     description: "Stanford professor of biology"},    │  │
  │  │    {name: "dopamine neurons",                         │  │
  │  │     entity_type: "NEURAL_SYSTEM",                     │  │
  │  │     description: "Neurons that produce dopamine"},    │  │
  │  │    {name: "wanting-liking gap",                       │  │
  │  │     entity_type: "CONCEPT",                           │  │
  │  │     description: "Distinction between desire..."}     │  │
  │  │  ]                                                    │  │
  │  │                                                       │  │
  │  │  relationships: [                                     │  │
  │  │    {source: "Sapolsky",                               │  │
  │  │     target: "dopamine neurons",                       │  │
  │  │     type: "EXPLAINS",                                 │  │
  │  │     description: "Describes the function of..."},     │  │
  │  │    {source: "dopamine neurons",                       │  │
  │  │     target: "wanting-liking gap",                     │  │
  │  │     type: "CREATES",                                  │  │
  │  │     description: "Anticipatory firing creates..."}    │  │
  │  │  ]                                                    │  │
  │  └───────────────────────────────────────────────────────┘  │
  │                                                             │
  │  Each entity/relationship tagged with source_chunk_id      │
  └─────────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                   TYPE CONSOLIDATION                        │
  │                                                             │
  │  Problem: LLM assigns types freely, creating duplicates:    │
  │  - "RESEARCHER", "SCIENTIST", "ACADEMIC" → Same concept     │
  │  - "BRAIN_REGION", "NEURAL_STRUCTURE" → Same concept        │
  │                                                             │
  │  Solution: LLM consolidation prompt                         │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  Input: 347 unique entity types discovered            │  │
  │  │  - RESEARCHER: 1,234 mentions                         │  │
  │  │  - SCIENTIST: 456 mentions                            │  │
  │  │  - ACADEMIC: 123 mentions                             │  │
  │  │  - BRAIN_REGION: 2,345 mentions                       │  │
  │  │  ...                                                  │  │
  │  │                                                       │  │
  │  │  Output: 15 consolidated types                        │  │
  │  │  - RESEARCHER (merges: SCIENTIST, ACADEMIC)           │  │
  │  │  - NEURAL_SYSTEM (merges: BRAIN_REGION, NEURAL_...)   │  │
  │  │  - CONCEPT, NEUROTRANSMITTER, BEHAVIOR, ...           │  │
  │  └───────────────────────────────────────────────────────┘  │
  │                                                             │
  │  Consolidation Strategies:                                  │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  stratified (default): Balances across corpora        │  │
  │  │                                                       │  │
  │  │  Corpus 1 (Neuroscience): Top types                   │  │
  │  │  - NEURAL_SYSTEM (45%), NEUROTRANSMITTER (12%)       │  │
  │  │                                                       │  │
  │  │  Corpus 2 (Philosophy): Top types                     │  │
  │  │  - PHILOSOPHER (38%), CONCEPT (22%)                   │  │
  │  │                                                       │  │
  │  │  → Final: Mix from both (prevents domain dominance)   │  │
  │  └───────────────────────────────────────────────────────┘  │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  global: Ranks by total count (single-domain corpora) │  │
  │  └───────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  OUTPUT FILES                                               │
  │                                                             │
  │  data/processed/07_graph/extraction_results.json            │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  {                                                    │  │
  │  │    "entities": [                                      │  │
  │  │      {"name": "dopamine", "entity_type": "NEURO...",  │  │
  │  │       "description": "...", "source_chunk_id": "..."},│  │
  │  │      ... (15,000+ entities)                           │  │
  │  │    ],                                                 │  │
  │  │    "relationships": [                                 │  │
  │  │      {"source": "dopamine", "target": "reward",       │  │
  │  │       "type": "REGULATES", ...},                      │  │
  │  │      ... (25,000+ relationships)                      │  │
  │  │    ]                                                  │  │
  │  │  }                                                    │  │
  │  └───────────────────────────────────────────────────────┘  │
  │                                                             │
  │  data/processed/07_graph/discovered_types.json              │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  {                                                    │  │
  │  │    "consolidated_entity_types": [                     │  │
  │  │      "RESEARCHER", "NEURAL_SYSTEM", "CONCEPT", ...    │  │
  │  │    ],                                                 │  │
  │  │    "consolidation_method": "stratified"               │  │
  │  │  }                                                    │  │
  │  └───────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────┘
```

**Code Location:** `src/graph/auto_tuning.py`

**Key Functions:**
- `extract_chunk()` - LLM extraction for single chunk
- `extract_book()` - Process all chunks in a book
- `consolidate_stratified()` - Balance types across corpora
- `consolidate_global()` - Rank by total frequency

---

### Stage 6b: Graph Construction & Community Detection

**Purpose:** Build Neo4j graph, run Leiden communities, generate summaries.

```
RUN: docker compose up -d neo4j
RUN: python -m src.stages.run_stage_6b_neo4j --embed-entities
```

#### Detailed Flow

```
            GRAPH CONSTRUCTION & COMMUNITY DETECTION
═════════════════════════════════════════════════════════════════════

  INPUT: extraction_results.json (15K entities, 25K relationships)

  ┌──────────────────────────────────────────────────────────────────┐
  │                 STEP 1: ENTITY NORMALIZATION                     │
  │                                                                  │
  │  Problem: Same entity appears with different surface forms:      │
  │  - "The Prefrontal Cortex", "prefrontal cortex", "PFC"          │
  │  - "Dopamine", "dopamine", "DA"                                 │
  │                                                                  │
  │  Solution: Normalize before Neo4j upload                        │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  normalized_name() algorithm (src/graph/schemas.py):       │ │
  │  │                                                            │ │
  │  │  1. NFKC Unicode normalization: "café" → "cafe"            │ │
  │  │  2. Lowercase: "Dopamine" → "dopamine"                     │ │
  │  │  3. Strip leading/trailing stopwords:                      │ │
  │  │     "the prefrontal cortex" → "prefrontal cortex"          │ │
  │  │  4. Remove punctuation: "dopamine," → "dopamine"           │ │
  │  │  5. Collapse whitespace: "  dopamine  " → "dopamine"       │ │
  │  │                                                            │ │
  │  │  Stopwords: {'the', 'a', 'an', 'of', 'and', 'or', ...}    │ │
  │  └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                 STEP 2: NEO4J UPLOAD (MERGE)                     │
  │                                                                  │
  │  Cypher Query (idempotent - safe to re-run):                    │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  MERGE (e:Entity {normalized_name: $normalized_name})      │ │
  │  │  ON CREATE SET                                             │ │
  │  │    e.name = $name,                                         │ │
  │  │    e.entity_type = $entity_type,                           │ │
  │  │    e.description = $description,                           │ │
  │  │    e.chunk_ids = [$source_chunk_id],                       │ │
  │  │    e.mention_count = 1                                     │ │
  │  │  ON MATCH SET                                              │ │
  │  │    e.chunk_ids = e.chunk_ids + $source_chunk_id,           │ │
  │  │    e.mention_count = e.mention_count + 1,                  │ │
  │  │    e.description = CASE                                    │ │
  │  │      WHEN size(e.description) < size($description)         │ │
  │  │      THEN $description ELSE e.description END              │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  Result:                                                        │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  (:Entity {                                                │ │
  │  │    name: "dopamine",                                       │ │
  │  │    normalized_name: "dopamine",                            │ │
  │  │    entity_type: "NEUROTRANSMITTER",                        │ │
  │  │    description: "A monoamine neurotransmitter...",         │ │
  │  │    chunk_ids: ["behave::chunk_42", "behave::chunk_100"],   │ │
  │  │    mention_count: 234                                      │ │
  │  │  })                                                        │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  Relationships:                                                 │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  MATCH (source:Entity {normalized_name: $source_norm})     │ │
  │  │  MATCH (target:Entity {normalized_name: $target_norm})     │ │
  │  │  MERGE (source)-[r:RELATED_TO]->(target)                   │ │
  │  │  SET r.type = $relationship_type,                          │ │
  │  │      r.description = $description,                         │ │
  │  │      r.weight = $weight,                                   │ │
  │  │      r.chunk_ids = coalesce(r.chunk_ids, []) + $chunk_id   │ │
  │  └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │             STEP 3: LEIDEN COMMUNITY DETECTION                   │
  │                                                                  │
  │  Graph Data Science (GDS) library provides the algorithm:       │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  # Project graph to GDS in-memory format                   │ │
  │  │  graph = gds.graph.project(                                │ │
  │  │    "graphrag",                                             │ │
  │  │    "Entity",                                               │ │
  │  │    {"RELATED_TO": {"orientation": "UNDIRECTED"}}           │ │
  │  │  )                                                         │ │
  │  │                                                            │ │
  │  │  # Run Leiden with DETERMINISTIC settings                  │ │
  │  │  result = gds.leiden.stream(                               │ │
  │  │    graph,                                                  │ │
  │  │    gamma=1.0,                    # Resolution parameter    │ │
  │  │    randomSeed=42,                # FIXED SEED              │ │
  │  │    concurrency=1,                # SINGLE THREAD           │ │
  │  │    includeIntermediateCommunities=True  # For hierarchy    │ │
  │  │  )                                                         │ │
  │  │                                                            │ │
  │  │  # Determinism guarantee:                                  │ │
  │  │  # Same graph + same seed + same concurrency               │ │
  │  │  # = IDENTICAL community assignments (guaranteed)          │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  Leiden vs Louvain:                                             │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  Louvain: May produce disconnected communities (broken)    │ │
  │  │  Leiden:  GUARANTEES connected communities (always valid)  │ │
  │  │                                                            │ │
  │  │  Leiden adds a "refinement phase" after each iteration    │ │
  │  │  that ensures communities stay connected.                  │ │
  │  └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │              STEP 4: HIERARCHICAL PARSING                        │
  │                                                                  │
  │  Leiden returns intermediateCommunityIds for each node:          │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  Node: "dopamine"                                          │ │
  │  │  communityId: 42                                           │ │
  │  │  intermediateCommunityIds: [5, 42]  ← [L1, L0]             │ │
  │  │                                                            │ │
  │  │  This means:                                               │ │
  │  │  - At Level 0 (finest): Community 42                       │ │
  │  │  - At Level 1 (coarser): Community 5                       │ │
  │  │  - At Level 2 (coarsest): Community 5 (no further split)   │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  parse_leiden_hierarchy() creates:                              │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  Level 0 (C0) - Finest granularity                         │ │
  │  │  ├── Community 42: {dopamine, serotonin, norepinephrine}   │ │
  │  │  ├── Community 43: {prefrontal cortex, amygdala}           │ │
  │  │  └── ... (7,000+ communities, 3-10 entities each)          │ │
  │  │                                                            │ │
  │  │  Level 1 (C1) - Medium granularity                         │ │
  │  │  ├── Community 5: {Comm 42, Comm 43, ...} → Neuroscience   │ │
  │  │  ├── Community 8: {Comm 100, Comm 101} → Philosophy        │ │
  │  │  └── ... (500-1000 communities)                            │ │
  │  │                                                            │ │
  │  │  Level 2 (C2) - Coarsest granularity                       │ │
  │  │  ├── Community 1: All science topics                       │ │
  │  │  └── Community 2: All philosophy topics                    │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  Parent-child relationships:                                    │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │                 ┌───────────────┐                          │ │
  │  │                 │   C2: Comm 1  │  (corpus-wide)           │ │
  │  │                 └───────┬───────┘                          │ │
  │  │                ┌────────┴────────┐                         │ │
  │  │         ┌──────┴──────┐   ┌──────┴──────┐                  │ │
  │  │         │ C1: Comm 5  │   │ C1: Comm 8  │  (domain-level)  │ │
  │  │         └──────┬──────┘   └──────┬──────┘                  │ │
  │  │         ┌──────┴──────┐   ┌──────┴──────┐                  │ │
  │  │   ┌─────┴─────┐ ┌─────┴─────┐  ┌─────┴─────┐               │ │
  │  │   │C0: Comm42 │ │C0: Comm43 │  │C0: Comm100│  (specific)   │ │
  │  │   └───────────┘ └───────────┘  └───────────┘               │ │
  │  └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │              STEP 5: PAGERANK CENTRALITY                         │
  │                                                                  │
  │  Purpose: Rank entities by importance (hub entities first)       │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  result = gds.pageRank.stream(                             │ │
  │  │    graph,                                                  │ │
  │  │    dampingFactor=0.85,    # Standard PageRank parameter    │ │
  │  │    maxIterations=20                                        │ │
  │  │  )                                                         │ │
  │  │                                                            │ │
  │  │  # Write scores back to Neo4j                              │ │
  │  │  SET e.pagerank = $score                                   │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  PageRank scores determine entity ordering in summaries:        │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  Community 42 entities (sorted by PageRank):               │ │
  │  │  1. dopamine (PR: 0.042) ← Most connected                  │ │
  │  │  2. reward system (PR: 0.031)                              │ │
  │  │  3. motivation (PR: 0.028)                                 │ │
  │  │  4. pleasure (PR: 0.019)                                   │ │
  │  │                                                            │ │
  │  │  High PageRank = Many connections = Hub entity             │ │
  │  └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │              STEP 6: COMMUNITY SUMMARIZATION                     │
  │                                                                  │
  │  For each community at each level:                              │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  1. Collect members (sorted by PageRank)                   │ │
  │  │  2. Collect internal relationships                         │ │
  │  │  3. Generate LLM summary                                   │ │
  │  │  4. Create embedding of summary                            │ │
  │  │  5. Upload to Weaviate (atomic, crash-proof)               │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  Community Context (input to LLM):                              │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  ## Entities                                               │ │
  │  │  - dopamine (NEUROTRANSMITTER) [PR:0.042] - A monoamine... │ │
  │  │  - reward system (NEURAL_SYSTEM) [PR:0.031] - Brain...     │ │
  │  │  - motivation (CONCEPT) [PR:0.028] - The drive to...       │ │
  │  │                                                            │ │
  │  │  ## Relationships                                          │ │
  │  │  - dopamine --[REGULATES]--> motivation: Controls the...   │ │
  │  │  - reward system --[INVOLVES]--> dopamine: Uses dopamine.. │ │
  │  │  - motivation --[DRIVES]--> behavior: Motivates action...  │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  LLM Prompt (GRAPHRAG_COMMUNITY_PROMPT):                        │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  "You are an expert summarizer. Given the following       │ │
  │  │   entities and relationships from a community in a        │ │
  │  │   knowledge graph, write a comprehensive summary that:    │ │
  │  │                                                            │ │
  │  │   1. Describes the main theme of this community           │ │
  │  │   2. Highlights key entities and their roles              │ │
  │  │   3. Explains important relationships                     │ │
  │  │                                                            │ │
  │  │   {community_context}"                                     │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  LLM Response (summary):                                        │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  "This community centers on the neuroscience of           │ │
  │  │   motivation and reward. Dopamine, a key neurotransmitter,│ │
  │  │   plays a central role in regulating motivation through   │ │
  │  │   the brain's reward system. The mesolimbic pathway       │ │
  │  │   connects the ventral tegmental area (VTA) to the        │ │
  │  │   nucleus accumbens, creating the neural substrate for    │ │
  │  │   wanting and anticipation of rewards. Key insights       │ │
  │  │   include the distinction between 'wanting' (dopamine-    │ │
  │  │   mediated anticipation) and 'liking' (actual pleasure)." │ │
  │  └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │              STEP 7: WEAVIATE STORAGE                            │
  │                                                                  │
  │  Two collections created:                                       │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  Entity_section800_v1                                      │ │
  │  │  ┌──────────────────────────────────────────────────────┐  │ │
  │  │  │  Properties:                                         │  │ │
  │  │  │  - entity_name: "dopamine"                           │  │ │
  │  │  │  - entity_type: "NEUROTRANSMITTER"                   │  │ │
  │  │  │  - description: "A monoamine neurotransmitter..."    │  │ │
  │  │  │                                                      │  │ │
  │  │  │  Vector: [0.23, 0.45, 0.12, ...] ← Embedding of desc│  │ │
  │  │  │                                                      │  │ │
  │  │  │  Used for: Query entity extraction (embedding-based) │  │ │
  │  │  └──────────────────────────────────────────────────────┘  │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  Community_section800_v1                                   │ │
  │  │  ┌──────────────────────────────────────────────────────┐  │ │
  │  │  │  Properties:                                         │  │ │
  │  │  │  - community_id: "community_L0_42"                   │  │ │
  │  │  │  - level: 0                                          │  │ │
  │  │  │  - summary: "This community centers on..."           │  │ │
  │  │  │  - member_count: 15                                  │  │ │
  │  │  │  - relationship_count: 23                            │  │ │
  │  │  │                                                      │  │ │
  │  │  │  Vector: [0.34, 0.56, 0.23, ...] ← Embedding of sum  │  │ │
  │  │  │                                                      │  │ │
  │  │  │  Used for: Community retrieval at query time         │  │ │
  │  │  └──────────────────────────────────────────────────────┘  │ │
  │  └────────────────────────────────────────────────────────────┘ │
  │                                                                  │
  │  Crash-Proof Design:                                            │
  │  ┌────────────────────────────────────────────────────────────┐ │
  │  │  1. Each community uploaded atomically                     │ │
  │  │  2. --resume flag checks Weaviate for existing IDs         │ │
  │  │  3. Deterministic Leiden = same IDs on re-run              │ │
  │  │  4. Can crash and resume without ID mismatches             │ │
  │  └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘
```

**Code Location:** `src/graph/community.py`, `src/graph/hierarchy.py`, `src/graph/centrality.py`

---

## Phase 2: Query Pipeline (Online)

### Step 1: Entity Extraction from Query

**Purpose:** Find relevant entities in the user's query to guide graph traversal.

```
                    QUERY ENTITY EXTRACTION
═══════════════════════════════════════════════════════════════════

  INPUT: User query "How does dopamine affect motivation?"

  ┌─────────────────────────────────────────────────────────────────┐
  │              PRIMARY: EMBEDDING-BASED EXTRACTION                │
  │              (~50ms, no LLM call required)                      │
  │                                                                 │
  │  1. Embed the query:                                           │
  │     ┌───────────────────────────────────────────────────────┐  │
  │     │  query = "How does dopamine affect motivation?"       │  │
  │     │         ↓ embed_texts([query])                        │  │
  │     │  query_embedding = [0.23, 0.45, 0.12, ...]            │  │
  │     └───────────────────────────────────────────────────────┘  │
  │                                                                 │
  │  2. Search Entity_section800_v1 collection:                    │
  │     ┌───────────────────────────────────────────────────────┐  │
  │     │  collection.query.near_vector(                        │  │
  │     │    query_embedding,                                   │  │
  │     │    limit=10,                    # top-k entities      │  │
  │     │    certainty=0.3                # min similarity      │  │
  │     │  )                                                    │  │
  │     │                                                       │  │
  │     │  Results (sorted by similarity):                      │  │
  │     │  1. dopamine (0.89)  ← High similarity                │  │
  │     │  2. motivation (0.85) ← High similarity               │  │
  │     │  3. reward (0.72)                                     │  │
  │     │  4. behavior (0.68)                                   │  │
  │     │  5. neurotransmitter (0.65)                           │  │
  │     └───────────────────────────────────────────────────────┘  │
  │                                                                 │
  │  3. Return entity names: ["dopamine", "motivation", "reward"]  │
  └─────────────────────────────────────────────────────────────────┘
                               │
                    If embedding returns empty
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │              FALLBACK 1: LLM-BASED EXTRACTION                   │
  │              (~1-2s, requires LLM call)                         │
  │                                                                 │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  Prompt (GRAPHRAG_QUERY_EXTRACTION_PROMPT):                │ │
  │  │                                                            │ │
  │  │  "Extract entity mentions from the following query.       │ │
  │  │   Return entities matching these types:                   │ │
  │  │   RESEARCHER, NEURAL_SYSTEM, CONCEPT, ...                 │ │
  │  │                                                            │ │
  │  │   Query: 'How does dopamine affect motivation?'           │ │
  │  │                                                            │ │
  │  │   Return JSON: {entities: [{name: '...'}]}"               │ │
  │  └───────────────────────────────────────────────────────────┘ │
  │                                                                 │
  │  LLM Response (QueryEntities):                                 │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  {"entities": [                                           │ │
  │  │    {"name": "dopamine"},                                  │ │
  │  │    {"name": "motivation"}                                 │ │
  │  │  ]}                                                       │ │
  │  └───────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────┘
                               │
                    If LLM returns empty
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │              FALLBACK 2: REGEX EXTRACTION                       │
  │                                                                 │
  │  Pattern: r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'              │
  │                                                                 │
  │  Finds capitalized words (proper nouns):                       │
  │  "What did Sapolsky say about stress?" → ["Sapolsky"]          │
  └─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │              NEO4J VALIDATION (optional)                        │
  │                                                                 │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  MATCH (e:Entity)                                         │ │
  │  │  WHERE e.normalized_name IN $entity_names                 │ │
  │  │  RETURN e.name, e.normalized_name                         │ │
  │  └───────────────────────────────────────────────────────────┘ │
  │                                                                 │
  │  Confirms entities exist in graph before traversal.            │
  │  Handles case differences: "Dopamine" matches "dopamine"       │
  └─────────────────────────────────────────────────────────────────┘

  OUTPUT: ["dopamine", "motivation"]
```

**Code Location:** `src/graph/query_entities.py`

---

### Step 2: Graph Traversal

**Purpose:** Find related entities and their source chunks by traversing the knowledge graph.

```
                       GRAPH TRAVERSAL
═══════════════════════════════════════════════════════════════════

  INPUT: Query entities ["dopamine", "motivation"]

  ┌─────────────────────────────────────────────────────────────────┐
  │              NEO4J 2-HOP TRAVERSAL                              │
  │                                                                 │
  │  For each query entity, find neighbors up to 2 hops away:      │
  │                                                                 │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  MATCH path = (start:Entity)-[*1..2]-(neighbor:Entity)    │ │
  │  │  WHERE start.normalized_name = 'dopamine'                 │ │
  │  │  RETURN                                                   │ │
  │  │    neighbor.name as name,                                 │ │
  │  │    neighbor.entity_type as entity_type,                   │ │
  │  │    neighbor.description as description,                   │ │
  │  │    neighbor.chunk_ids as source_chunk_ids,                │ │
  │  │    length(path) as path_length                            │ │
  │  │  ORDER BY path_length ASC, neighbor.pagerank DESC         │ │
  │  │  LIMIT 20                                                 │ │
  │  └───────────────────────────────────────────────────────────┘ │
  │                                                                 │
  │  Visual representation:                                        │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │                                                           │ │
  │  │                  ┌──────────────┐                         │ │
  │  │                  │   dopamine   │ ← START                 │ │
  │  │                  └──────┬───────┘                         │ │
  │  │           ┌─────────────┼─────────────┐                   │ │
  │  │           │             │             │                   │ │
  │  │           ▼             ▼             ▼                   │ │
  │  │   ┌──────────────┐ ┌──────────┐ ┌──────────────┐          │ │
  │  │   │reward system │ │serotonin │ │ motivation   │ 1-HOP    │ │
  │  │   └──────┬───────┘ └────┬─────┘ └──────────────┘          │ │
  │  │          │              │                                 │ │
  │  │          ▼              ▼                                 │ │
  │  │   ┌──────────────┐ ┌──────────────┐                       │ │
  │  │   │   pleasure   │ │   anxiety    │              2-HOP    │ │
  │  │   └──────────────┘ └──────────────┘                       │ │
  │  │                                                           │ │
  │  └───────────────────────────────────────────────────────────┘ │
  │                                                                 │
  │  Results (with path_length):                                   │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  1. reward system  (path_length=1, chunk_ids: [chunk_42]) │ │
  │  │  2. serotonin      (path_length=1, chunk_ids: [chunk_43]) │ │
  │  │  3. motivation     (path_length=1, chunk_ids: [chunk_100])│ │
  │  │  4. pleasure       (path_length=2, chunk_ids: [chunk_150])│ │
  │  │  5. anxiety        (path_length=2, chunk_ids: [chunk_200])│ │
  │  └───────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │              EXTRACT CHUNK IDS                                  │
  │                                                                 │
  │  From graph traversal results, collect unique chunk IDs:        │
  │                                                                 │
  │  graph_chunk_ids = [                                           │
  │    "behave::chunk_42",    ← reward system mentioned here       │
  │    "behave::chunk_43",    ← serotonin mentioned here           │
  │    "behave::chunk_100",   ← motivation mentioned here          │
  │    "behave::chunk_150",   ← pleasure mentioned here            │
  │    "behave::chunk_200",   ← anxiety mentioned here             │
  │  ]                                                             │
  │                                                                 │
  │  These chunks are relevant because they contain entities       │
  │  that are related to the query entities in the knowledge graph.│
  └─────────────────────────────────────────────────────────────────┘

  OUTPUT:
  - graph_chunk_ids: ["chunk_42", "chunk_43", "chunk_100", ...]
  - graph_context: [{name, path_length, source_chunk_id}, ...]
```

**Code Location:** `src/graph/neo4j_client.py` (find_entity_neighbors), `src/graph/query.py` (retrieve_graph_context)

---

### Step 3: RRF Merging

**Purpose:** Combine vector search and graph traversal results, boosting chunks that appear in both.

```
                       RRF (RECIPROCAL RANK FUSION)
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │                    THE RRF FORMULA                              │
  │                                                                 │
  │   RRF_score(chunk) = Σ  1 / (k + rank_i)                       │
  │                      i                                          │
  │                                                                 │
  │   Where:                                                        │
  │   - k = 60 (constant, prevents high-ranked items from          │
  │             dominating too much)                                │
  │   - rank_i = position in result list i (1-indexed)             │
  │   - i = each result list (vector, graph)                       │
  │                                                                 │
  │   KEY INSIGHT: Chunks appearing in BOTH lists get contributions│
  │   from both ranks, resulting in HIGHER combined scores.        │
  └─────────────────────────────────────────────────────────────────┘

  INPUTS:

  ┌─────────────────────────────────────────────────────────────────┐
  │   VECTOR RESULTS (ranked by embedding similarity)              │
  │                                                                 │
  │   Rank 1: chunk_1  (similarity: 0.92)                          │
  │   Rank 2: chunk_7  (similarity: 0.89)                          │
  │   Rank 3: chunk_12 (similarity: 0.87)  ◀── OVERLAP             │
  │   Rank 4: chunk_23 (similarity: 0.85)                          │
  │   Rank 5: chunk_42 (similarity: 0.83)  ◀── OVERLAP             │
  │   Rank 6: chunk_55 (similarity: 0.81)                          │
  │   ...                                                          │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │   GRAPH RESULTS (ranked by path_length, shorter = better)      │
  │                                                                 │
  │   Rank 1: chunk_42  (path_length: 1)  ◀── OVERLAP              │
  │   Rank 2: chunk_12  (path_length: 1)  ◀── OVERLAP              │
  │   Rank 3: chunk_100 (path_length: 2)                           │
  │   Rank 4: chunk_150 (path_length: 2)                           │
  │   ...                                                          │
  └─────────────────────────────────────────────────────────────────┘

  RRF CALCULATION:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   chunk_1:  (vector only)                                      │
  │     score = 1/(60+1) + 0 = 0.0164                              │
  │                                                                 │
  │   chunk_7:  (vector only)                                      │
  │     score = 1/(60+2) + 0 = 0.0161                              │
  │                                                                 │
  │   chunk_12: (BOTH lists - gets BOOSTED)                        │
  │     score = 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320     │
  │              ↑ vector     ↑ graph                               │
  │                                                                 │
  │   chunk_42: (BOTH lists - gets BOOSTED)                        │
  │     score = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318     │
  │              ↑ vector     ↑ graph                               │
  │                                                                 │
  │   chunk_100: (graph only)                                      │
  │     score = 0 + 1/(60+3) = 0.0159                              │
  │                                                                 │
  │   chunk_150: (graph only)                                      │
  │     score = 0 + 1/(60+4) = 0.0156                              │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  FINAL MERGED RANKING:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   1. chunk_12 (0.0320) ★ BOOSTED - in both lists               │
  │   2. chunk_42 (0.0318) ★ BOOSTED - in both lists               │
  │   3. chunk_1  (0.0164) - vector only (was #1 in vector)        │
  │   4. chunk_7  (0.0161) - vector only                           │
  │   5. chunk_100 (0.0159) - graph only                           │
  │   6. chunk_150 (0.0156) - graph only                           │
  │   ...                                                          │
  │                                                                 │
  │   EFFECT: Chunks found by BOTH semantic similarity AND         │
  │   knowledge graph relationships rise to the top.               │
  │                                                                 │
  │   This is the key insight of hybrid retrieval:                 │
  │   - Vector search finds semantically similar content           │
  │   - Graph traversal finds structurally related content         │
  │   - Overlap indicates HIGH RELEVANCE → boost it                │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

**Code Location:** `src/rag_pipeline/retrieval/rrf.py`, `src/graph/query.py` (hybrid_graph_retrieval)

---

### Step 4: Map-Reduce for Global Queries

**Purpose:** Answer thematic questions that span the entire corpus using community summaries.

```
                    MAP-REDUCE FOR GLOBAL QUERIES
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │              QUERY CLASSIFICATION                               │
  │                                                                 │
  │  Local Query (entity-focused):                                 │
  │  - "What is dopamine?"                                         │
  │  - "How does the prefrontal cortex work?"                      │
  │  - "What did Sapolsky say about stress?"                       │
  │  → Use: Entity traversal + RRF merge                           │
  │                                                                 │
  │  Global Query (thematic):                                      │
  │  - "What are the main themes across all books?"                │
  │  - "How do the authors' perspectives differ on free will?"     │
  │  - "What common patterns emerge across the corpus?"            │
  │  → Use: Map-reduce over community summaries                    │
  │                                                                 │
  │  Classification logic (should_use_map_reduce):                 │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  1. If entities were extracted → likely LOCAL             │ │
  │  │  2. If no entities extracted → classify with LLM          │ │
  │  │     Prompt: "Is this query local or global?"              │ │
  │  └───────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────┘

  GLOBAL QUERY: "What are the main themes across all 19 books?"

  ┌─────────────────────────────────────────────────────────────────┐
  │              STEP 1: RETRIEVE TOP-K COMMUNITIES                 │
  │                                                                 │
  │  Select highest-level communities (C2 preferred) for broad     │
  │  coverage. Use embedding similarity to find most relevant.     │
  │                                                                 │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  query_embedding = embed("What are the main themes...")   │ │
  │  │                                                           │ │
  │  │  Top-5 communities by similarity:                         │ │
  │  │  1. Community L2_1: "Neuroscience of behavior..."        │ │
  │  │  2. Community L2_2: "Philosophy of mind..."              │ │
  │  │  3. Community L1_5: "Reward and motivation..."           │ │
  │  │  4. Community L1_8: "Decision-making..."                 │ │
  │  │  5. Community L1_12: "Consciousness and free will..."    │ │
  │  └───────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │              STEP 2: MAP PHASE (PARALLEL)                       │
  │                                                                 │
  │  For each community, generate a partial answer:                │
  │                                                                 │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  Community 1            Community 2            ...      │   │
  │  │       │                      │                          │   │
  │  │       ▼                      ▼                          │   │
  │  │  ┌─────────┐           ┌─────────┐                      │   │
  │  │  │   LLM   │           │   LLM   │    (async parallel)  │   │
  │  │  └────┬────┘           └────┬────┘                      │   │
  │  │       │                      │                          │   │
  │  │       ▼                      ▼                          │   │
  │  │  "From a neuro-       "From a philosophical            │   │
  │  │   science view,        perspective, the                │   │
  │  │   the main themes      corpus explores                 │   │
  │  │   include..."          questions of..."                │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                 │
  │  MAP PROMPT (GRAPHRAG_MAP_PROMPT):                             │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  "Given the following community summary, answer the       │ │
  │  │   question based ONLY on information in this community.   │ │
  │  │   If not relevant, say 'Not relevant'.                   │ │
  │  │                                                           │ │
  │  │   Community Summary: {community_summary}                  │ │
  │  │   Top Entities: {top_entities}                            │ │
  │  │   Relationships: {relationships}                          │ │
  │  │                                                           │ │
  │  │   Question: {query}"                                      │ │
  │  └───────────────────────────────────────────────────────────┘ │
  │                                                                 │
  │  Partial answers:                                              │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  Community 1: "From a neuroscience perspective, the main  │ │
  │  │   themes include the role of dopamine in reward and       │ │
  │  │   motivation, the influence of stress on behavior, and    │ │
  │  │   the neural basis of decision-making..."                 │ │
  │  │                                                           │ │
  │  │  Community 2: "Philosophically, the corpus explores       │ │
  │  │   questions of free will, consciousness, and moral        │ │
  │  │   responsibility in light of neuroscientific findings..." │ │
  │  │                                                           │ │
  │  │  Community 3: "The theme of reward and motivation emerges │ │
  │  │   repeatedly, with emphasis on dopamine's role in both    │ │
  │  │   wanting and addiction..."                               │ │
  │  │                                                           │ │
  │  │  Community 4: "Decision-making is analyzed from multiple  │ │
  │  │   angles: economic (rational choice), psychological       │ │
  │  │   (biases), and neural (prefrontal cortex)..."            │ │
  │  │                                                           │ │
  │  │  Community 5: "The nature of consciousness and its        │ │
  │  │   relationship to physical brain processes is a central   │ │
  │  │   theme, with debates about determinism..."               │ │
  │  └───────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │              STEP 3: REDUCE PHASE                               │
  │                                                                 │
  │  Synthesize all partial answers into a coherent final answer:  │
  │                                                                 │
  │  REDUCE PROMPT (GRAPHRAG_REDUCE_PROMPT):                       │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  "Synthesize the following partial answers into a         │ │
  │  │   comprehensive response. Remove redundancy, highlight    │ │
  │  │   key themes, and create a coherent narrative.           │ │
  │  │                                                           │ │
  │  │   Question: {query}                                       │ │
  │  │                                                           │ │
  │  │   Partial Answers:                                        │ │
  │  │   Community 1: {partial_1}                                │ │
  │  │   Community 2: {partial_2}                                │ │
  │  │   ..."                                                    │ │
  │  └───────────────────────────────────────────────────────────┘ │
  │                                                                 │
  │  FINAL SYNTHESIZED ANSWER:                                     │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  "The 19 books explore several interconnected themes:     │ │
  │  │                                                           │ │
  │  │   1. **Reward and Motivation**: Dopamine's central role   │ │
  │  │      in the brain's reward circuitry is a recurring       │ │
  │  │      topic, from Sapolsky's analysis of wanting vs liking │ │
  │  │      to discussions of addiction mechanisms.              │ │
  │  │                                                           │ │
  │  │   2. **Free Will and Determinism**: Multiple authors      │ │
  │  │      grapple with how neuroscientific findings about      │ │
  │  │      decision-making challenge traditional notions of     │ │
  │  │      free will, with perspectives ranging from hard       │ │
  │  │      determinism to compatibilism.                        │ │
  │  │                                                           │ │
  │  │   3. **Stress and Behavior**: The impact of chronic       │ │
  │  │      stress on brain structure and function, particularly │ │
  │  │      the prefrontal cortex, appears across both           │ │
  │  │      scientific and philosophical discussions.            │ │
  │  │                                                           │ │
  │  │   4. **Consciousness**: The 'hard problem' of             │ │
  │  │      consciousness and its neural correlates forms        │ │
  │  │      a bridge between neuroscience and philosophy..."     │ │
  │  └───────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────┘

  TIMING (example):
  - Map phase: 5 communities × ~1s each = ~1s (parallel)
  - Reduce phase: ~2s
  - Total: ~3s (vs ~5s sequential)
```

**Code Location:** `src/graph/map_reduce.py`

---

## Key Algorithms Explained

### Entity Normalization

```python
# src/graph/schemas.py

def normalized_name(self) -> str:
    """Normalize entity name for consistent matching."""
    name = unicodedata.normalize('NFKC', self.name.strip())  # café → cafe
    name = name.lower()                                       # Dopamine → dopamine

    # Remove leading/trailing stopwords
    words = name.split()
    while words and words[0] in EDGE_STOPWORDS:  # {'the', 'a', 'an', ...}
        words.pop(0)
    while words and words[-1] in EDGE_STOPWORDS:
        words.pop()

    name = re.sub(r'[^\w\s]', '', ' '.join(words))  # Remove punctuation
    return ' '.join(name.split())                    # Collapse whitespace
```

### RRF Score Calculation

```python
# src/rag_pipeline/retrieval/rrf.py

def reciprocal_rank_fusion(result_lists, query_types, k=60, top_k=10):
    """Merge result lists using Reciprocal Rank Fusion."""
    scores = defaultdict(float)

    for results in result_lists:
        for rank, result in enumerate(results):
            # Core RRF formula: 1/(k + rank)
            # rank is 0-indexed, so add 1
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.chunk_id] += rrf_score

    # Chunks appearing in multiple lists get summed scores → BOOSTED
    sorted_chunks = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_chunks[:top_k]
```

### Leiden Hierarchy Parsing

```python
# src/graph/hierarchy.py

def parse_leiden_hierarchy(leiden_result, max_levels=3):
    """Parse Leiden intermediate communities into hierarchy levels."""

    for node_community in leiden_result["node_communities"]:
        node_id = node_community["node_id"]
        intermediate_ids = node_community["intermediate_ids"]

        # intermediate_ids is coarse-to-fine in Neo4j GDS
        # Reverse to get fine-to-coarse (L0, L1, L2)
        hierarchy = list(reversed(intermediate_ids))

        # Assign node to communities at each level
        for level_idx in range(max_levels):
            community_id = hierarchy[level_idx]
            levels[level_idx].communities[community_id].add(node_id)

            # Track parent-child relationships
            if level_idx + 1 < len(hierarchy):
                parent_id = hierarchy[level_idx + 1]
                levels[level_idx].parent_map[community_id] = parent_id
```

---

## Configuration Reference

All GraphRAG settings in `src/config.py`:

<div align="center">

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRAPHRAG_EXTRACTION_MODEL` | `anthropic/claude-3-haiku` | LLM for entity extraction |
| `GRAPHRAG_MAX_ENTITIES` | `10` | Max entities per chunk |
| `GRAPHRAG_MAX_RELATIONSHIPS` | `7` | Max relationships per chunk |
| `GRAPHRAG_TYPES_PER_CORPUS` | `12` | Entity types per corpus (stratified) |
| `GRAPHRAG_MIN_CORPUS_PERCENTAGE` | `1.0` | Min % for type consolidation |
| `GRAPHRAG_LEIDEN_RESOLUTION` | `1.0` | Higher = more, smaller communities |
| `GRAPHRAG_LEIDEN_SEED` | `42` | Random seed (deterministic) |
| `GRAPHRAG_LEIDEN_CONCURRENCY` | `1` | Thread count (1 for reproducibility) |
| `GRAPHRAG_MIN_COMMUNITY_SIZE` | `3` | Min entities per community |
| `GRAPHRAG_MAX_HIERARCHY_LEVELS` | `3` | Community levels (C0, C1, C2) |
| `GRAPHRAG_PAGERANK_DAMPING` | `0.85` | Standard PageRank damping |
| `GRAPHRAG_TRAVERSE_DEPTH` | `2` | Graph traversal hops |
| `GRAPHRAG_TOP_COMMUNITIES` | `3` | Communities in context |
| `GRAPHRAG_RRF_K` | `60` | RRF constant |
| `GRAPHRAG_ENTITY_EXTRACTION_TOP_K` | `10` | Query entities (embedding) |
| `GRAPHRAG_ENTITY_MIN_SIMILARITY` | `0.3` | Min entity similarity |
| `GRAPHRAG_MAP_REDUCE_TOP_K` | `5` | Communities for map-reduce |
| `GRAPHRAG_MAP_MAX_TOKENS` | `300` | Tokens per map response |
| `GRAPHRAG_REDUCE_MAX_TOKENS` | `500` | Tokens for reduce response |

</div>

---

## Running the Pipeline

### Prerequisites

```bash
# Activate environment
conda activate raglab

# Start services
docker compose up -d neo4j weaviate

# Verify connections
python -c "from src.graph import verify_connection; verify_connection()"
```

### Indexing (One-Time, ~10 hours)

```bash
# Step 1: Entity extraction with auto-tuning
python -m src.stages.run_stage_4_5_autotune --strategy section

# Step 1b: Re-consolidate for mixed corpora (optional)
python -m src.stages.run_stage_4_5_autotune --reconsolidate stratified

# Step 2: Upload + Leiden + PageRank + Summaries + Entity Embeddings
python -m src.stages.run_stage_6b_neo4j --embed-entities

# Resume after crash
python -m src.stages.run_stage_6b_neo4j --resume
```

### Querying

```bash
# CLI evaluation
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag --search-type hybrid

# UI: Select "graphrag" in preprocessing dropdown
```

---

## Performance & Cost Analysis

### Indexing Costs (One-Time)

<div align="center">

| Phase | LLM Calls | Cost | Time |
|-------|-----------|------|------|
| Entity extraction | ~5,000 | ~$3-5 | ~5h |
| Type consolidation | 1 | ~$0.01 | <1m |
| Community summarization | ~7,000 | ~$2-3 | ~5h |
| Entity embeddings | ~15,000 | ~$0.50 | ~30m |
| **Total** | ~27,000 | ~$5-8 | ~10h |

</div>

### Query Costs (Per Request)

<div align="center">

| Component | Method | Latency |
|-----------|--------|---------|
| Entity extraction | Embedding search | ~50ms |
| Entity extraction | LLM fallback | ~1-2s |
| Graph traversal | Neo4j Cypher | ~100ms |
| Community retrieval | Weaviate HNSW | ~50ms |
| RRF merge | Python | ~10ms |
| **Total (local)** | | ~200ms-2s |
| **Total (global + map-reduce)** | | ~3-5s |

</div>

### Key Performance Metrics

From comprehensive evaluation (102 configurations):

<div align="center">

| Metric | GraphRAG | Baseline (None) |
|--------|----------|-----------------|
| Cross-domain correctness | **50.1%** (+5%) | 47.7% |
| Single-concept recall | **97.5%** | 92.3% |

</div>

---

## Navigation

**Next:** [Reranking](reranking.md) — Cross-encoder for precision

**Related:**
- [RAPTOR](../chunking/raptor.md) — Alternative hierarchy via clustering
- [HyDE](hyde.md) — Simpler cross-domain (no Neo4j)
- [Query Decomposition](query-decomposition.md) — Sub-query strategy
- [Preprocessing Overview](README.md) — Strategy comparison
