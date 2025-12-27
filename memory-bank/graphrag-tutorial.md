# GraphRAG: Complete Technical Deep-Dive for RAG1-Mini

A comprehensive tutorial covering the theory, implementation, configuration, data structures, algorithms, and execution of GraphRAG in this project.

---

## Table of Contents
1. [The Problem GraphRAG Solves](#1-the-problem-graphrag-solves)
2. [Core Concepts & Architecture](#2-core-concepts--architecture)
3. [Configuration Deep-Dive](#3-configuration-deep-dive)
4. [Data Structures & Schemas](#4-data-structures--schemas)
5. [Auto-Tuning Entity Type Discovery](#5-auto-tuning-entity-type-discovery)
6. [Entity Extraction Pipeline](#6-entity-extraction-pipeline)
7. [Knowledge Graph Construction (Neo4j)](#7-knowledge-graph-construction-neo4j)
8. [Leiden Community Detection Algorithm](#8-leiden-community-detection-algorithm)
9. [Hybrid Retrieval at Query Time](#9-hybrid-retrieval-at-query-time)
10. [Execution Guide](#10-execution-guide)
11. [Neo4j Browser & Cypher Queries](#11-neo4j-browser--cypher-queries)
12. [Data Flow Trace](#12-data-flow-trace)
13. [RAPTOR vs GraphRAG Comparison](#13-raptor-vs-graphrag-comparison)
14. [GraphRAG Research Updates (2025)](#14-graphrag-research-updates-2025)
15. [Sources](#15-sources)

---

## 1. The Problem GraphRAG Solves

### Standard RAG Limitation

Standard RAG answers questions by finding chunks semantically similar to the query. This works for **local queries** (specific facts) but fails for **global queries** that require synthesizing information across documents.

```
STANDARD RAG FAILURE CASE
═════════════════════════

Query: "What are the main themes across all books in the corpus?"

┌─────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING SPACE                                 │
│                          (flat, unstructured)                           │
│                                                                         │
│    [chunk_1: "Dopamine affects motivation..."]                         │
│                                        *                                │
│    [chunk_2: "Marcus Aurelius wrote..."]                               │
│          *                                                              │
│    [chunk_3: "The prefrontal cortex..."]                               │
│                    *                                                    │
│                                                                         │
│           [QUERY: "main themes?"]                                       │
│                    *                                                    │
│                                                                         │
│    PROBLEM: Query is too abstract to match any specific chunk!         │
│    Returns: Random chunks that happen to mention "themes" or "main"    │
└─────────────────────────────────────────────────────────────────────────┘
```

### GraphRAG's Solution: Structure + Summarization

GraphRAG builds a **knowledge graph** that captures entities and relationships, then clusters them into **communities** with LLM-generated summaries.

```
GRAPHRAG APPROACH
═════════════════

Query: "What are the main themes across all books?"

┌─────────────────────────────────────────────────────────────────────────┐
│                      KNOWLEDGE GRAPH                                    │
│                                                                         │
│    [Marcus Aurelius]────AUTHORED────>[Meditations]                      │
│           │                               │                             │
│       PRACTICED                       TEACHES                           │
│           │                               │                             │
│           v                               v                             │
│      [Stoicism]──────────────────>[Self-Control]                        │
│           │                               │                             │
│       RELATES_TO                      MODULATES                         │
│           │                               │                             │
│           v                               v                             │
│     [Acceptance]                  [Prefrontal Cortex]                   │
│                                          │                              │
│                                      REGULATES                          │
│                                          │                              │
│                                          v                              │
│                                    [Emotions]                           │
│                                                                         │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ COMMUNITY 1: "Stoic Philosophy & Self-Regulation"              │   │
│    │ Summary: "This community covers Stoic practices for           │   │
│    │ emotional regulation, connecting Marcus Aurelius's teachings  │   │
│    │ with neuroscience research on prefrontal cortex control..."   │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│    Response: Uses community summaries to synthesize corpus themes!     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quantitative Results (from Microsoft paper)

| Metric | GraphRAG vs Standard RAG |
|--------|--------------------------|
| Comprehensiveness | 72-83% win rate |
| Answer Diversity | 62-82% win rate |
| Token Efficiency | 97% fewer tokens at query time |
| Context Precision | +70% improvement |

---

## 2. Core Concepts & Architecture

### The Two-Phase Architecture

GraphRAG operates in two distinct phases:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        INDEXING PHASE (Offline)                           ║
║                     Runs once when documents are added                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ STEP 1: TEXT CHUNKING (Stage 4)                                     │ ║
║   │                                                                     │ ║
║   │   data/processed/05_final_chunks/section/*.json                     │ ║
║   │                                                                     │ ║
║   │   [Chunk 1: 800 tokens]  [Chunk 2: 800 tokens]  [Chunk 3: ...]     │ ║
║   │   Each chunk has: text, chunk_id, book_name, context_path           │ ║
║   └───────────────────────────────────┬─────────────────────────────────┘ ║
║                                       │                                   ║
║                                       ▼                                   ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ STEP 2: ENTITY & RELATIONSHIP EXTRACTION (Stage 4.6)                │ ║
║   │                                                                     │ ║
║   │   For each chunk, LLM extracts:                                     │ ║
║   │   - Entities: [name, type, description, source_chunk_id]            │ ║
║   │   - Relationships: [source, target, type, description, weight]      │ ║
║   │                                                                     │ ║
║   │   Uses structured output (Pydantic → JSON Schema)                   │ ║
║   │   Model: anthropic/claude-3-haiku (cost-effective)                  │ ║
║   │                                                                     │ ║
║   │   Output: data/processed/05_final_chunks/graph/extraction_results.json           │ ║
║   └───────────────────────────────────┬─────────────────────────────────┘ ║
║                                       │                                   ║
║                                       ▼                                   ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ STEP 3: KNOWLEDGE GRAPH CONSTRUCTION (Stage 6b)                     │ ║
║   │                                                                     │ ║
║   │   Upload to Neo4j:                                                  │ ║
║   │   - Entity nodes: MERGE on normalized_name (deduplication)          │ ║
║   │   - RELATED_TO edges: MERGE with type property                      │ ║
║   │                                                                     │ ║
║   │   Neo4j browser: http://localhost:7474                              │ ║
║   └───────────────────────────────────┬─────────────────────────────────┘ ║
║                                       │                                   ║
║                                       ▼                                   ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ STEP 4: LEIDEN COMMUNITY DETECTION (Stage 6b)                       │ ║
║   │                                                                     │ ║
║   │   Algorithm: Leiden (improvement over Louvain)                      │ ║
║   │   Purpose: Group densely-connected entities into clusters           │ ║
║   │   Output: community_id property on each Entity node                 │ ║
║   └───────────────────────────────────┬─────────────────────────────────┘ ║
║                                       │                                   ║
║                                       ▼                                   ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ STEP 5: COMMUNITY SUMMARIZATION (Stage 6b)                          │ ║
║   │                                                                     │ ║
║   │   For each community (size >= 3):                                   │ ║
║   │   1. Get members and internal relationships                         │ ║
║   │   2. Format as context string                                       │ ║
║   │   3. LLM generates thematic summary                                 │ ║
║   │                                                                     │ ║
║   │   Output: data/processed/05_final_chunks/graph/communities.json                  │ ║
║   └─────────────────────────────────────────────────────────────────────┘ ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════╗
║                         QUERY PHASE (Online)                              ║
║                        Runs for each user query                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ User Query: "How does Sapolsky explain dopamine's role in stress?" │ ║
║   └───────────────────────────────────┬─────────────────────────────────┘ ║
║                                       │                                   ║
║           ┌───────────────────────────┼───────────────────────┐           ║
║           │                           │                       │           ║
║           ▼                           ▼                       ▼           ║
║   ┌───────────────┐         ┌─────────────────┐       ┌───────────────┐   ║
║   │ Vector Search │         │ Graph Traversal │       │ Community     │   ║
║   │ (Weaviate)    │         │ (Neo4j)         │       │ Matching      │   ║
║   │               │         │                 │       │               │   ║
║   │ Semantic      │         │ 1. Extract:     │       │ Keyword match │   ║
║   │ similarity    │         │ [Sapolsky,      │       │ on community  │   ║
║   │ on query      │         │  dopamine,      │       │ summaries     │   ║
║   │ embedding     │         │  stress]        │       │               │   ║
║   │               │         │                 │       │ Returns top-3 │   ║
║   │ Returns:      │         │ 2. Traverse     │       │ summaries     │   ║
║   │ top-k chunks  │         │ 2 hops from     │       │               │   ║
║   │ by score      │         │ each entity     │       │               │   ║
║   │               │         │                 │       │               │   ║
║   │               │         │ 3. Collect      │       │               │   ║
║   │               │         │ source chunks   │       │               │   ║
║   └───────┬───────┘         └────────┬────────┘       └───────┬───────┘   ║
║           │                          │                        │           ║
║           └──────────────────────────┼────────────────────────┘           ║
║                                      │                                    ║
║                                      ▼                                    ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ HYBRID MERGE                                                        │ ║
║   │                                                                     │ ║
║   │ 1. Mark vector results that also appear in graph as "boosted"       │ ║
║   │ 2. Sort: boosted results first, then remaining by score             │ ║
║   │ 3. Include community summaries as thematic context                  │ ║
║   │                                                                     │ ║
║   │ Result: [chunk_12★, chunk_45★, chunk_78, chunk_33, ...]            │ ║
║   │         (★ = graph-boosted)                                         │ ║
║   └───────────────────────────────────┬─────────────────────────────────┘ ║
║                                       │                                   ║
║                                       ▼                                   ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ ANSWER GENERATION                                                   │ ║
║   │                                                                     │ ║
║   │ Context includes:                                                   │ ║
║   │ - Retrieved chunks (graph-boosted first)                            │ ║
║   │ - Community summaries (thematic context)                            │ ║
║   │ - Entity relationships (from graph)                                 │ ║
║   │                                                                     │ ║
║   │ LLM generates comprehensive answer                                  │ ║
║   └─────────────────────────────────────────────────────────────────────┘ ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Two Extraction Paths: Choose One

**IMPORTANT:** There are two ways to run entity extraction. Both produce the same output file (`extraction_results.json`), but differ in how entity types are determined:

| Stage | Command | How Entity Types Are Determined |
|-------|---------|--------------------------------|
| **Stage 4.5 autotune** | `python -m src.stages.run_stage_4_5_autotune` | Discovered from your corpus content |
| **Stage 4.6 graph_extract** | `python -m src.stages.run_stage_4_6_graph_extract` | Hardcoded in `src/config.py` |

**When to use each:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DECISION TREE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Is this a new domain (not neuroscience/philosophy)?                       │
│       │                                                                     │
│       ├── YES → Use 4.5 autotune (discovers types from YOUR content)       │
│       │         - More accurate for domain-specific documents               │
│       │         - Takes longer (extra LLM calls for type consolidation)     │
│       │         - Creates discovered_types.json for future reference        │
│       │                                                                     │
│       └── NO → Either works, but 4.5 autotune still recommended            │
│                 - 4.6 uses predefined types that may not match your corpus  │
│                 - 4.5 adapts to your actual content                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Execution flow:**

```
Stage 4 (chunks) ─────────────────────────────────────────────────► Stage 6b
                       │                                                │
                       ▼                                                ▼
             ┌─────────────────────┐                           Neo4j Upload
             │ Choose ONE:         │                                +
             │                     │                           Leiden Detection
             │ • 4.5 autotune      │──► extraction_results.json     +
             │   (recommended)     │                           Community Summaries
             │                     │
             │ • 4.6 graph_extract │
             └─────────────────────┘
```

**Output files from extraction:**

| File | Created By | Purpose |
|------|------------|---------|
| `data/processed/07_graph/extraction_results.json` | Both 4.5 and 4.6 | Entities + relationships for Neo4j |
| `data/processed/07_graph/discovered_types.json` | Only 4.5 autotune | Auto-discovered entity/relationship types |

---

## 3. Configuration Deep-Dive

All GraphRAG settings are in `src/config.py` (lines 554-669):

### Neo4j Connection Settings

```python
# src/config.py, lines 562-564

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rag1mini_graphrag")
```

**What these mean:**
- `bolt://localhost:7687`: Neo4j's binary protocol (faster than HTTP)
- Credentials must match `docker-compose.yml` line 115: `NEO4J_AUTH: neo4j/rag1mini_graphrag`

### Entity Extraction Settings

```python
# src/config.py, lines 567-598

# Model for extraction (reuses contextual chunking model for consistency)
GRAPHRAG_EXTRACTION_MODEL = "anthropic/claude-3-haiku"  # ~$0.25/1M tokens

# Max tokens for extraction response
GRAPHRAG_MAX_EXTRACTION_TOKENS = 2000

# Domain-specific entity types (guides LLM extraction)
GRAPHRAG_ENTITY_TYPES = [
    # Neuroscience domain
    "BRAIN_REGION",          # prefrontal cortex, amygdala, hippocampus
    "NEUROTRANSMITTER",      # dopamine, serotonin, cortisol
    "NEURAL_PROCESS",        # synaptic plasticity, long-term potentiation
    "COGNITIVE_FUNCTION",    # working memory, decision-making, attention
    "BEHAVIOR",              # aggression, altruism, stress response

    # Philosophy domain
    "PHILOSOPHER",           # Marcus Aurelius, Schopenhauer, Confucius
    "PHILOSOPHICAL_CONCEPT", # virtue ethics, will, Tao, Stoic acceptance
    "PHILOSOPHICAL_SCHOOL",  # Stoicism, Taoism, German Pessimism
    "TEXT_OR_WORK",          # Meditations, The Art of Worldly Wisdom

    # Research domain
    "RESEARCHER",            # Sapolsky, Kahneman, Tversky
    "STUDY_OR_EXPERIMENT",   # Stanford prison experiment, marshmallow test
    "COGNITIVE_BIAS",        # confirmation bias, loss aversion

    # General
    "PERSON",                # historical figures, case study subjects
    "ORGANIZATION",          # universities, research institutions
    "BOOK_OR_CHAPTER",       # source document references
]
```

**Why domain-specific types matter:**
- Generic NER might extract "Stanford" as ORGANIZATION when it's part of "Stanford prison experiment"
- Domain types guide LLM to extract contextually relevant entities
- Types become searchable properties in Neo4j

### Relationship Types

```python
# src/config.py, lines 601-624

GRAPHRAG_RELATIONSHIP_TYPES = [
    # Causal/mechanistic
    "CAUSES",                # A causes B
    "INHIBITS",              # A inhibits/blocks B
    "MODULATES",             # A modulates/affects B
    "REGULATES",             # A regulates B

    # Associative
    "ASSOCIATED_WITH",       # A is associated with B
    "PART_OF",               # A is part of B
    "LOCATED_IN",            # A is located in B

    # Philosophical
    "PROPOSES",              # Philosopher proposes concept
    "INFLUENCES",            # A influences B
    "CONTRADICTS",           # A contradicts B
    "BUILDS_ON",             # A builds on B
    "ADVOCATES_FOR",         # A advocates for B

    # Research
    "STUDIES",               # Researcher studies phenomenon
    "DEMONSTRATES",          # Study demonstrates finding
    "CITES",                 # A cites B

    # Attribution
    "AUTHORED_BY",           # Work authored by person
    "AFFILIATED_WITH",       # Person affiliated with organization
]
```

### Leiden Algorithm Parameters

```python
# src/config.py, lines 627-629

GRAPHRAG_LEIDEN_RESOLUTION = 1.0    # Higher = more, smaller communities
GRAPHRAG_LEIDEN_MAX_LEVELS = 10     # Maximum hierarchy depth
GRAPHRAG_MIN_COMMUNITY_SIZE = 3     # Skip tiny communities
```

**Resolution parameter explained:**
- Resolution γ = 1.0: Standard modularity optimization
- Resolution γ > 1.0: Favors smaller communities (more granular)
- Resolution γ < 1.0: Favors larger communities (more merged)

### Community Summarization

```python
# src/config.py, lines 632-647

GRAPHRAG_MAX_SUMMARY_TOKENS = 200   # Output limit per summary
GRAPHRAG_MAX_CONTEXT_TOKENS = 6000  # Input context limit

GRAPHRAG_COMMUNITY_PROMPT = """You are analyzing a community of related entities...
1. Identifies the main theme or topic connecting these entities
2. Explains the key relationships and how concepts interact
3. Highlights important details, names, and specific findings

Summary:"""
```

### Query-Time Retrieval Parameters

```python
# src/config.py, lines 664-666

GRAPHRAG_TOP_COMMUNITIES = 3        # Communities to retrieve
GRAPHRAG_TRAVERSE_DEPTH = 2         # Max hops from query entities
GRAPHRAG_RRF_K = 60                 # RRF constant (not currently used)
```

**Traverse depth explained:**
- Depth 1: Direct neighbors only (conservative)
- Depth 2: Neighbors of neighbors (recommended)
- Depth 3+: May include loosely related entities

---

## 4. Data Structures & Schemas

All schemas are defined in `src/graph/schemas.py` using Pydantic for JSON Schema generation and validation.

### GraphEntity Schema

```python
# src/graph/schemas.py, lines 27-78

class GraphEntity(BaseModel):
    """Single entity extracted from text."""

    name: str = Field(
        ...,  # Required
        description="The entity name as it appears in the text",
        min_length=1,
    )
    entity_type: str = Field(
        ...,
        description="Entity type from the allowed types list",
    )
    description: str = Field(
        default="",
        description="Brief description of this entity in context (1-2 sentences)",
    )
    source_chunk_id: str = Field(
        default="",
        description="Chunk ID where this entity was extracted from",
    )

    def normalized_name(self) -> str:
        """For deduplication: 'Dopamine' -> 'dopamine'"""
        return self.name.strip().lower()
```

**Example entity object:**
```json
{
  "name": "prefrontal cortex",
  "entity_type": "BRAIN_REGION",
  "description": "Brain region involved in executive function and impulse control",
  "source_chunk_id": "behave::chunk_42"
}
```

### GraphRelationship Schema

```python
# src/graph/schemas.py, lines 80-137

class GraphRelationship(BaseModel):
    """Relationship between two entities."""

    source_entity: str = Field(
        ...,
        description="Name of the source entity (from)",
    )
    target_entity: str = Field(
        ...,
        description="Name of the target entity (to)",
    )
    relationship_type: str = Field(
        ...,
        description="Relationship type from the allowed types list",
    )
    description: str = Field(
        default="",
        description="Brief description of this relationship (1 sentence)",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence or strength of relationship (0.0-1.0)",
    )
    source_chunk_id: str = Field(
        default="",
        description="Chunk ID where this relationship was extracted from",
    )
```

**Example relationship object:**
```json
{
  "source_entity": "dopamine",
  "target_entity": "reward system",
  "relationship_type": "MODULATES",
  "description": "Dopamine modulates reward processing in the brain",
  "weight": 1.0,
  "source_chunk_id": "behave::chunk_42"
}
```

### ExtractionResult Schema (LLM Output)

```python
# src/graph/schemas.py, lines 139-167

class ExtractionResult(BaseModel):
    """Result of entity/relationship extraction from a single chunk.

    This model is passed to call_structured_completion() which
    generates a JSON Schema for the LLM to follow.
    """

    entities: List[GraphEntity] = Field(
        default_factory=list,
        description="List of entities extracted from the text",
    )
    relationships: List[GraphRelationship] = Field(
        default_factory=list,
        description="List of relationships between entities",
    )
```

**How JSON Schema enforcement works:**
```
1. Pydantic model → JSON Schema (via model_json_schema())
2. Schema passed to OpenRouter API as response_format
3. LLM generates valid JSON matching the schema
4. Pydantic validates and parses the response
```

### Community Schema

```python
# src/graph/schemas.py, lines 191-242

class Community(BaseModel):
    """Leiden community with members and summary."""

    community_id: str        # "community_42"
    level: int = 0           # Hierarchy level (0 = finest)
    members: List[CommunityMember]
    member_count: int
    relationship_count: int
    summary: str             # LLM-generated theme description
    embedding: Optional[List[float]] = None  # For future vector search
```

**Example community object:**
```json
{
  "community_id": "community_42",
  "level": 0,
  "member_count": 8,
  "relationship_count": 12,
  "summary": "This community focuses on the neurobiology of stress...",
  "members": [
    {
      "entity_name": "cortisol",
      "entity_type": "NEUROTRANSMITTER",
      "description": "Stress hormone released by adrenal glands",
      "degree": 6
    },
    ...
  ]
}
```

### Data Storage Locations

```
data/processed/05_final_chunks/graph/
├── extraction_results.json    # Stage 4.6 output
│   {
│     "entities": [...],       # All extracted entities
│     "relationships": [...],  # All extracted relationships
│     "stats": {
│       "total_chunks": 150,
│       "processed_chunks": 148,
│       "failed_chunks": 2,
│       "total_entities": 450,
│       "total_relationships": 320,
│       "unique_entity_types": 12,
│       "unique_relationship_types": 8
│     }
│   }
│
└── communities.json           # Stage 6b output
    {
      "communities": [...],    # List of Community objects
      "total_count": 12,
      "total_members": 180
    }
```

---

## 5. Auto-Tuning Entity Type Discovery

### The Problem with Manual Entity Types

Standard GraphRAG requires manually defining entity types (BRAIN_REGION, PHILOSOPHER, etc.). This has issues:
- Entity types may not match actual corpus content
- Different domains need different types
- Query-time entity extraction may miss concepts not in predefined list

### Auto-Tuning Solution

Auto-tuning discovers entity types from the actual corpus:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTO-TUNING PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: OPEN-ENDED EXTRACTION                                              │
│  ═══════════════════════════════                                             │
│                                                                             │
│  For each book (atomic unit):                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Prompt: "Extract entities and relationships from this text.          │   │
│  │          Assign MOST APPROPRIATE TYPE for each entity.                │   │
│  │          You may create NEW types if needed."                        │   │
│  │                                                                       │   │
│  │ LLM freely assigns: CONCEPT, COGNITIVE_PROCESS, NEURAL_STRUCTURE,   │   │
│  │                     RESEARCHER, DISCIPLINE, BEHAVIOR, etc.           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: graph/extractions/{book}.json (per-book, atomic)                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 2: TYPE AGGREGATION                                                   │
│  ═════════════════════════                                                   │
│                                                                             │
│  Merge all per-book extractions and count types:                           │
│                                                                             │
│  Entity Types Discovered:                                                   │
│    CONCEPT: 38 occurrences                                                 │
│    COGNITIVE_PROCESS: 30 occurrences                                       │
│    NEURAL_STRUCTURE: 10 occurrences                                        │
│    RESEARCHER: 10 occurrences                                              │
│    ...                                                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 3: LLM CONSOLIDATION                                                  │
│  ═════════════════════════                                                   │
│                                                                             │
│  LLM consolidates similar types into clean taxonomy:                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Prompt: "Analyze these discovered types. Consolidate similar types,  │   │
│  │          remove types with count=1, propose clean taxonomy of         │   │
│  │          15-25 entity types and 10-20 relationship types."           │   │
│  │                                                                       │   │
│  │ Result:                                                               │   │
│  │   BRAIN_REGION + NEURAL_STRUCTURE → NEURAL_STRUCTURE                 │   │
│  │   FIELD_OF_STUDY (count=1) → removed                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: graph/discovered_types.json                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Per-Book Resumable Processing

Auto-tuning processes each book atomically with resume support:

```python
# src/graph/auto_tuning.py

def run_auto_tuning_resumable(
    strategy: str = "section",
    overwrite_context: Optional[OverwriteContext] = None,
    ...
) -> Dict[str, Any]:
    """Run auto-tuning with per-book resume support."""

    book_files = load_book_files(strategy)  # 19 books
    extractions_dir = DIR_GRAPH_DATA / "extractions"

    for book_path in book_files:
        book_name = book_path.stem
        output_path = extractions_dir / f"{book_name}.json"

        # Check overwrite decision (prompt/skip/all)
        if not overwrite_context.should_overwrite(output_path, logger):
            continue  # Skip already processed

        # Extract from this book (may take 5-20 minutes)
        results = extract_book(book_path, model=model)

        # Save atomically - if interrupted here, book not marked complete
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Merge all per-book results
    merged = merge_book_extractions(extractions_dir)

    # Consolidate types with LLM
    if not skip_consolidation:
        consolidated = consolidate_types(merged["entity_type_counts"], ...)
```

### Output Files

```
data/processed/05_final_chunks/graph/
├── extractions/                          # Per-book results (atomic)
│   ├── Behave, The_Biology of Humans.json
│   ├── Biopsychology.json
│   ├── Brain and behavior.json
│   └── ... (16 more)
├── extraction_results.json               # Merged from all books
└── discovered_types.json                 # Consolidated taxonomy

data/logs/
└── autotune_20251226_143000.log          # Execution log
```

### discovered_types.json Structure

```json
{
  "consolidated_entity_types": [
    "CONCEPT",
    "COGNITIVE_PROCESS",
    "NEURAL_STRUCTURE",
    "RESEARCHER",
    "DISCIPLINE",
    "BEHAVIOR",
    "TECHNOLOGY",
    "DISORDER"
  ],
  "consolidated_relationship_types": [
    "STUDIES",
    "OPENED_DOORS_TO",
    "STUDIED_BY",
    "CONTAINS",
    "REPRESENTS",
    "ALLOWS_RECOGNITION_OF",
    "INCLUDES",
    "ENABLES",
    "INVOLVES",
    "PART_OF",
    "ABSENT_IN",
    "INFLUENCES",
    "PROPOSES",
    "PROVIDES_INPUT_TO",
    "DECREASES_DURING"
  ],
  "consolidation_rationale": "The entity types have been consolidated by merging similar types...",
  "raw_entity_type_counts": {
    "CONCEPT": 38,
    "COGNITIVE_PROCESS": 30,
    ...
  }
}
```

### CLI Usage

```bash
# Full auto-tuning (all 19 books, ~2.5 hours)
python -m src.stages.run_stage_4_5_autotune

# Resume after interruption
python -m src.stages.run_stage_4_5_autotune --overwrite skip

# Force reprocess all books
python -m src.stages.run_stage_4_5_autotune --overwrite all

# Preview books to process
python -m src.stages.run_stage_4_5_autotune --list-books

# Show discovered types
python -m src.stages.run_stage_4_5_autotune --show-types
```

### Integration with Query-Time Extraction

The discovered types are used at query time for LLM-based entity extraction:

```python
# src/graph/query.py

def _get_entity_types() -> List[str]:
    """Get entity types, preferring discovered types if available."""
    discovered_path = DIR_GRAPH_DATA / "discovered_types.json"
    if discovered_path.exists():
        with open(discovered_path, "r") as f:
            data = json.load(f)
        return data["consolidated_entity_types"]
    else:
        return GRAPHRAG_ENTITY_TYPES  # Fallback to predefined
```

This ensures query-time extraction uses the same vocabulary as indexed entities.

---

## 6. Entity Extraction Pipeline

### The Extraction Prompt

```python
# src/config.py, lines 649-662

GRAPHRAG_EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Entity types to look for: {entity_types}
Relationship types to look for: {relationship_types}

Text:
{text}

Extract all entities and relationships following the JSON schema provided.
Be thorough but precise - only extract entities that are explicitly mentioned.
For relationships, only include those that are clearly stated or strongly implied."""
```

### Extraction Flow (src/graph/extractor.py)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENTITY EXTRACTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: Load Chunks (lines 174-219)                                   │
│  ─────────────────────────────────────                                 │
│  load_chunks_for_extraction(strategy="section")                        │
│                                                                         │
│  Reads: data/processed/05_final_chunks/section/*.json                  │
│  Returns: List[Dict] with text, chunk_id, context fields               │
│                                                                         │
│  Example chunk:                                                         │
│  {                                                                      │
│    "text": "The prefrontal cortex, located at the front of the brain...",│
│    "chunk_id": "behave::chunk_42",                                     │
│    "book_name": "behave",                                              │
│    "context_path": "Chapter 2 > The Biology of Self-Control"          │
│  }                                                                      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 2: Extract from Each Chunk (lines 49-100)                        │
│  ──────────────────────────────────────────────────                     │
│  extract_from_chunk(chunk, model="anthropic/claude-3-haiku")           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Build prompt:                                                    │   │
│  │                                                                  │   │
│  │ prompt = GRAPHRAG_EXTRACTION_PROMPT.format(                     │   │
│  │     entity_types=", ".join(GRAPHRAG_ENTITY_TYPES),              │   │
│  │     relationship_types=", ".join(GRAPHRAG_RELATIONSHIP_TYPES),  │   │
│  │     text=chunk["text"],                                          │   │
│  │ )                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                             │
│                          ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Call LLM with structured output:                                 │   │
│  │                                                                  │   │
│  │ result = call_structured_completion(                            │   │
│  │     messages=[{"role": "user", "content": prompt}],             │   │
│  │     model="anthropic/claude-3-haiku",                           │   │
│  │     response_model=ExtractionResult,  # Pydantic schema         │   │
│  │     temperature=0.0,                  # Deterministic            │   │
│  │     max_tokens=2000,                                             │   │
│  │ )                                                                 │   │
│  │                                                                  │   │
│  │ OpenRouter automatically enforces JSON Schema from Pydantic     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                             │
│                          ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Add source tracking:                                             │   │
│  │                                                                  │   │
│  │ for entity in result.entities:                                   │   │
│  │     entity.source_chunk_id = chunk["chunk_id"]                  │   │
│  │                                                                  │   │
│  │ for rel in result.relationships:                                 │   │
│  │     rel.source_chunk_id = chunk["chunk_id"]                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 3: Aggregate Results (lines 103-171)                             │
│  ─────────────────────────────────────────────                         │
│  extract_from_chunks(chunks, model, max_chunks)                        │
│                                                                         │
│  for i, chunk in enumerate(chunks):                                    │
│      result = extract_from_chunk(chunk)                                │
│      all_entities.extend(result.entities)                              │
│      all_relationships.extend(result.relationships)                    │
│                                                                         │
│      # Progress logging every 10 chunks                                │
│      if (i + 1) % 10 == 0:                                             │
│          logger.info(f"Processed {i+1}/{len(chunks)} chunks")          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 4: Save Results (lines 222-246)                                  │
│  ─────────────────────────────────────                                 │
│  save_extraction_results(results)                                      │
│                                                                         │
│  Output: data/processed/05_final_chunks/graph/extraction_results.json               │
│                                                                         │
│  {                                                                      │
│    "entities": [                                                        │
│      {"name": "prefrontal cortex", "entity_type": "BRAIN_REGION", ...},│
│      {"name": "dopamine", "entity_type": "NEUROTRANSMITTER", ...},     │
│    ],                                                                   │
│    "relationships": [                                                   │
│      {"source_entity": "prefrontal cortex", "target_entity": "...", ...}│
│    ],                                                                   │
│    "stats": {...}                                                       │
│  }                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Knowledge Graph Construction (Neo4j)

### Docker Configuration

```yaml
# docker-compose.yml, lines 94-135

neo4j:
  image: neo4j:5.26.0-enterprise    # Enterprise for GDS plugin
  container_name: neo4j_rag

  ports:
    - "7474:7474"   # Browser UI (http://localhost:7474)
    - "7687:7687"   # Bolt protocol (Python driver)

  volumes:
    - ./neo4j_data:/data        # Persistent storage
    - ./neo4j_logs:/logs        # Log files
    - ./neo4j_plugins:/plugins  # GDS plugin

  environment:
    NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"   # Required for Enterprise
    NEO4J_AUTH: neo4j/rag1mini_graphrag     # Credentials
    NEO4J_PLUGINS: '["graph-data-science"]' # Enable GDS (Leiden)

    # Memory settings
    NEO4J_dbms_memory_heap_initial__size: 512m
    NEO4J_dbms_memory_heap_max__size: 1G
    NEO4J_dbms_memory_pagecache_size: 512m
```

### Graph Schema in Neo4j

```
NODE LABELS
═══════════

(:Entity)
    - name: String              # "dopamine"
    - normalized_name: String   # "dopamine" (lowercase for MERGE)
    - entity_type: String       # "NEUROTRANSMITTER"
    - description: String       # "Brain chemical involved in reward"
    - source_chunk_id: String   # "behave::chunk_42"
    - community_id: Integer     # 42 (after Leiden)
    - created_at: DateTime      # Timestamp


RELATIONSHIP TYPES
══════════════════

(:Entity)-[:RELATED_TO]->(:Entity)
    - type: String              # "MODULATES"
    - description: String       # "Dopamine modulates reward processing"
    - weight: Float             # 1.0
    - source_chunk_id: String   # "behave::chunk_42"
    - created_at: DateTime      # Timestamp


INDEXES (created automatically)
════════════════════════════════

entity_name:        ON Entity(normalized_name)  # For MERGE deduplication
entity_type:        ON Entity(entity_type)      # For filtering
entity_chunk:       ON Entity(source_chunk_id)  # For source tracking
```

### Entity Upload with Deduplication

```python
# src/graph/neo4j_client.py, lines 191-245

def upload_entities(driver, entities, batch_size=100):
    """Upload entities using MERGE for deduplication."""

    query = """
    UNWIND $entities AS entity
    MERGE (e:Entity {normalized_name: toLower(trim(entity.name))})
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
```

**How MERGE deduplication works:**
```
Chunk 1: "Dopamine affects motivation"
  → Entity: {name: "Dopamine", entity_type: "NEUROTRANSMITTER"}
  → MERGE creates new node

Chunk 2: "dopamine is a neurotransmitter"
  → Entity: {name: "dopamine", entity_type: "NEUROTRANSMITTER"}
  → normalized_name = "dopamine" matches existing node
  → MERGE updates existing node (keeps longer description)

Result: Single node for "dopamine" with best description
```

---

## 8. Leiden Community Detection Algorithm

### What is Leiden?

Leiden is a community detection algorithm that groups densely-connected nodes into clusters. It improves on Louvain by guaranteeing well-connected communities.

```
LEIDEN ALGORITHM OVERVIEW
═════════════════════════

Input: Undirected graph G = (V, E)
Output: Partition P of nodes into communities

PHASE 1: LOCAL MOVING
──────────────────────
- Each node starts in its own community
- Nodes move to neighboring communities if it improves modularity
- Modularity Q measures: (edges within communities) vs (expected by chance)

    Before:            After:
    [A]  [B]  [C]      ┌─────────┐
     │    │    │       │ [A][B]  │  Community 1
     └────┼────┘       └─────────┘
          │            ┌─────────┐
         [D]           │ [C][D]  │  Community 2
                       └─────────┘


PHASE 2: REFINEMENT (Leiden improvement over Louvain)
─────────────────────────────────────────────────────
- Check each community is internally well-connected
- Split communities that have disconnected subparts
- Louvain could produce badly-connected communities; Leiden prevents this


PHASE 3: AGGREGATION
────────────────────
- Treat each community as a single super-node
- Apply algorithm recursively
- Build hierarchy of communities

    Level 0 (finest):    [1] [2] [3] [4] [5] [6]
                           \  |  /     \  |  /
    Level 1:             [  Comm A  ] [  Comm B  ]
                              \           /
    Level 2 (coarsest):    [     Root Comm     ]


RESOLUTION PARAMETER (γ)
────────────────────────
γ = 1.0: Standard modularity (default)
γ > 1.0: More, smaller communities
γ < 1.0: Fewer, larger communities
```

### Leiden in Neo4j GDS

```python
# src/graph/community.py, lines 53-153

def project_graph(gds, graph_name="graphrag"):
    """Create in-memory graph projection for GDS algorithms."""

    # Drop existing projection if exists
    if gds.graph.exists(graph_name).exists:
        gds.graph.drop(graph_name)

    # Project Entity nodes and RELATED_TO relationships
    graph, result = gds.graph.project(
        graph_name,
        "Entity",  # Node label to project
        {
            "RELATED_TO": {
                "orientation": "UNDIRECTED",  # Leiden requires undirected
                "properties": ["weight"],     # Edge weights for quality
            }
        },
        nodeProperties=["entity_type"],  # Node properties to include
    )

    return graph


def run_leiden(gds, graph, resolution=1.0, max_levels=10):
    """Run Leiden community detection."""

    result = gds.leiden.stream(
        graph,
        gamma=resolution,                    # Resolution parameter
        maxLevels=max_levels,                # Hierarchy depth
        includeIntermediateCommunities=True, # Get full hierarchy
    )

    # Convert to list of dicts
    node_communities = []
    for record in result.itertuples():
        node_communities.append({
            "node_id": record.nodeId,
            "community_id": record.communityId,
        })

    return {
        "community_count": len(set(nc["community_id"] for nc in node_communities)),
        "node_communities": node_communities,
    }
```

### Community Summarization

```python
# src/graph/community.py, lines 252-329

def build_community_context(members, relationships, max_tokens=6000):
    """Format community data for LLM summarization."""

    lines = []

    # Entities section
    lines.append("## Entities")
    for member in members:
        desc = f" - {member.description}" if member.description else ""
        lines.append(f"- {member.entity_name} ({member.entity_type}){desc}")

    # Relationships section
    if relationships:
        lines.append("\n## Relationships")
        for rel in relationships:
            desc = f": {rel['description']}" if rel.get("description") else ""
            lines.append(
                f"- {rel['source']} --[{rel['relationship_type']}]--> {rel['target']}{desc}"
            )

    return "\n".join(lines)


def summarize_community(members, relationships, model):
    """Generate LLM summary for a community."""

    context = build_community_context(members, relationships)

    prompt = GRAPHRAG_COMMUNITY_PROMPT.format(community_context=context)

    summary = call_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,  # Some creativity for narrative
        max_tokens=200,
    )

    return summary.strip()
```

**Example community context sent to LLM:**
```markdown
## Entities
- dopamine (NEUROTRANSMITTER) - Brain chemical involved in reward and motivation
- reward system (BRAIN_REGION) - Neural circuitry processing rewards
- prefrontal cortex (BRAIN_REGION) - Executive function and impulse control
- Robert Sapolsky (RESEARCHER) - Stanford neuroscientist studying behavior

## Relationships
- dopamine --[MODULATES]--> reward system: Dopamine modulates reward processing
- prefrontal cortex --[REGULATES]--> reward system: PFC regulates reward responses
- Robert Sapolsky --[STUDIES]--> dopamine: Sapolsky researches dopamine's behavioral effects
```

**Example LLM-generated summary:**
```
This community centers on the neurobiology of reward and motivation. Dopamine,
a key neurotransmitter, modulates the reward system, which is in turn regulated
by the prefrontal cortex. Robert Sapolsky's research at Stanford has extensively
examined how dopamine influences behavior, particularly in the context of
decision-making and impulse control. The community highlights the interplay
between brain chemistry and higher cognitive functions.
```

---

## 9. Hybrid Retrieval at Query Time

### Entity Extraction from Query

```python
# src/graph/query.py, lines 46-92

def extract_query_entities(query, driver=None):
    """Extract potential entity mentions from query.

    Pattern 1: Capitalized words (proper nouns)
    Pattern 2: Neo4j lookup for known entities
    """

    entities = []

    # Pattern 1: Regex for capitalized words
    cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    capitalized = re.findall(cap_pattern, query)
    entities.extend(capitalized)

    # Pattern 2: Check Neo4j for matches
    if driver and entities:
        db_entities = find_entities_by_names(driver, entities)
        entities.extend([e["name"] for e in db_entities])

    # Deduplicate
    return list(dict.fromkeys(entities))
```

**Example:**
```
Query: "How does Sapolsky explain dopamine's role in stress?"

Extracted entities: ["Sapolsky", "dopamine", "stress"]
  - "Sapolsky" from capitalization pattern
  - Verified in Neo4j as RESEARCHER
```

### Graph Traversal

```python
# src/graph/neo4j_client.py, lines 378-416

def find_entity_neighbors(driver, entity_name, max_hops=2, limit=50):
    """Find entities connected within N hops."""

    query = f"""
    MATCH (start:Entity {{normalized_name: toLower(trim($entity_name))}})
    MATCH path = (start)-[*1..{max_hops}]-(neighbor:Entity)
    WHERE start <> neighbor
    RETURN DISTINCT
        neighbor.name as name,
        neighbor.entity_type as entity_type,
        neighbor.description as description,
        neighbor.source_chunk_id as source_chunk_id,
        length(path) as path_length
    ORDER BY path_length, name
    LIMIT $limit
    """

    result = driver.execute_query(query, entity_name=entity_name, limit=limit)
    return [dict(r) for r in result.records]
```

**Example traversal:**
```
Start: "dopamine"
Depth: 2

┌─────────────┐
│  dopamine   │
└──────┬──────┘
       │ MODULATES
       ▼
┌─────────────────┐
│  reward system  │────PART_OF────►[mesolimbic pathway]
└────────┬────────┘
         │ REGULATES
         ▼
┌─────────────────┐
│ prefrontal      │────MODULATES────►[decision-making]
│ cortex          │
└─────────────────┘

Collected chunk IDs from all visited entities
```

### Hybrid Merge Logic

```python
# src/graph/query.py, lines 330-399

def hybrid_graph_retrieval(query, driver, vector_results, top_k=10):
    """Merge vector search results with graph traversal."""

    # 1. Get graph chunk IDs via traversal
    graph_chunk_ids, graph_meta = get_graph_chunk_ids(query, driver)

    # 2. Get community context for thematic enrichment
    community_context = retrieve_community_context(query)

    # 3. Mark vector results that also appear in graph
    graph_set = set(graph_chunk_ids)
    for result in vector_results:
        if result.get("chunk_id") in graph_set:
            result["graph_boost"] = True

    # 4. Sort: boosted first, then by original score
    boosted = [r for r in vector_results if r.get("graph_boost")]
    non_boosted = [r for r in vector_results if not r.get("graph_boost")]

    merged = boosted + non_boosted

    return merged[:top_k], {
        "query_entities": graph_meta["query_entities"],
        "graph_context": graph_meta["graph_context"],
        "community_context": community_context,
        "boosted_count": len(boosted),
    }
```

**Merge example:**
```
Vector Search Results          Graph Chunk IDs
═══════════════════            ═══════════════
1. chunk_12 (score: 0.85)      chunk_12 ← Match!
2. chunk_45 (score: 0.82)      chunk_33
3. chunk_78 (score: 0.79)      chunk_67
4. chunk_23 (score: 0.75)      chunk_12


After Hybrid Merge:
══════════════════
1. chunk_12 ★ (boosted - in both)
2. chunk_45
3. chunk_78
4. chunk_23

★ = Found via both vector search AND graph traversal
```

---

## 10. Execution Guide

### Prerequisites

```bash
# 1. Activate environment
conda activate rag1-mini

# 2. Start Docker services
docker compose up -d

# 3. Verify services are running
docker compose ps
# Should show: weaviate_rag (8080, 50051) and neo4j_rag (7474, 7687)

# 4. Ensure baseline chunks exist (if not already done)
python -m src.stages.run_stage_1_extraction   # PDF -> Markdown
python -m src.stages.run_stage_2_processing   # Clean Markdown
python -m src.stages.run_stage_3_segmentation # Sentence segmentation
python -m src.stages.run_stage_4_chunking     # Create chunks
```

### Stage 4.5: Auto-Tuning (Optional but Recommended)

```bash
# Auto-discover entity types from corpus (all 19 books, ~2.5 hours)
python -m src.stages.run_stage_4_5_autotune

# Resume after interruption (skip completed books)
python -m src.stages.run_stage_4_5_autotune --overwrite skip

# Preview books to process
python -m src.stages.run_stage_4_5_autotune --list-books

# Show discovered types
python -m src.stages.run_stage_4_5_autotune --show-types
```

**Expected output:**
```
============================================================
Stage 4.5: Auto-Tune Entity Types (Resumable)
============================================================
Log file: data/logs/autotune_20251226_143000.log
Overwrite mode: prompt
Model: anthropic/claude-3-haiku
Found 19 books to process
Extracting from Behave, The_Biology of Humans: 450 chunks
  [Behave] 20/450 chunks, 85 entities
  [Behave] 40/450 chunks, 162 entities
  ...
Saved: Behave, The_Biology of Humans.json
...
============================================================
Auto-Tuning Complete
============================================================
Books processed: 19
Books skipped: 0
Total chunks: 6249
Total entities: 1250
Unique entity types: 8
```

### Stage 4.6: Entity Extraction

```bash
# Full extraction (all chunks)
python -m src.stages.run_stage_4_6_graph_extract

# Test with subset (faster)
python -m src.stages.run_stage_4_6_graph_extract --max-chunks 20

# Use specific chunking strategy
python -m src.stages.run_stage_4_6_graph_extract --strategy contextual

# Override model
python -m src.stages.run_stage_4_6_graph_extract --model anthropic/claude-3-opus
```

**Expected output:**
```
============================================================
STAGE 4.6: GRAPHRAG ENTITY EXTRACTION
============================================================
Strategy: section
Model: anthropic/claude-3-haiku
Loading chunks from section strategy...
Found 150 chunks to process
Extracting entities from 150 chunks...
Processed 10/150 chunks, 45 entities, 32 relationships
Processed 20/150 chunks, 92 entities, 68 relationships
...
------------------------------------------------------------
EXTRACTION COMPLETE
------------------------------------------------------------
Chunks processed: 148
Entities extracted: 450
Relationships extracted: 320
Unique entity types: 12
Unique relationship types: 8
Failed chunks: 2
Time elapsed: 180.5s
Output: data/processed/05_final_chunks/graph/extraction_results.json
```

### Stage 6b: Neo4j Upload + Leiden

```bash
# Full pipeline (upload + Leiden + summarization)
python -m src.stages.run_stage_6b_neo4j

# Upload only (skip Leiden)
python -m src.stages.run_stage_6b_neo4j --upload-only

# Leiden only (graph already exists)
python -m src.stages.run_stage_6b_neo4j --leiden-only

# Clear graph before upload (fresh start)
python -m src.stages.run_stage_6b_neo4j --clear

# Override summarization model
python -m src.stages.run_stage_6b_neo4j --model anthropic/claude-3-opus
```

**Expected output:**
```
============================================================
STAGE 6b: NEO4J UPLOAD + LEIDEN COMMUNITIES
============================================================
Connected to Neo4j at bolt://localhost:7687
------------------------------------------------------------
PHASE 1: UPLOAD TO NEO4J
------------------------------------------------------------
Loaded 450 entities, 320 relationships
Creating indexes...
Uploaded 450 entities to Neo4j
Uploaded 320 relationships to Neo4j
Upload complete in 5.2s
  Entities: 450
  Relationships: 320
Graph stats: 450 nodes, 320 relationships
------------------------------------------------------------
PHASE 2: LEIDEN COMMUNITY DETECTION
------------------------------------------------------------
Dropped existing graph projection: graphrag
Projected graph 'graphrag': 450 nodes, 320 relationships
Leiden found 12 communities across 450 nodes
Updated 450 nodes with community IDs
Summarizing community 0 (15 members, 28 relationships)
Summarizing community 1 (12 members, 18 relationships)
...
Saved 12 communities to data/processed/05_final_chunks/graph/communities.json
Leiden complete in 45.3s
  Communities found: 12
  Total members: 180
============================================================
STAGE 6b COMPLETE
============================================================
Total time: 50.5s
```

### Query via UI

```bash
# Launch Streamlit
streamlit run src/ui/app.py

# Open browser: http://localhost:8501
# 1. Select "GraphRAG" from preprocessing strategy dropdown
# 2. Enter query
# 3. View Pipeline Log for graph metadata
```

### Evaluation

```bash
# Single run with GraphRAG
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag

# Comprehensive grid search (all strategies)
python -m src.stages.run_stage_7_evaluation --comprehensive
```

---

## 11. Neo4j Browser & Cypher Queries

### Accessing Neo4j Browser

1. Open: **http://localhost:7474**
2. Login:
   - Connect URL: `neo4j://localhost:7687`
   - Username: `neo4j`
   - Password: `rag1mini_graphrag`

### Essential Cypher Queries

```cypher
-- View all entity types and counts
MATCH (e:Entity)
RETURN e.entity_type AS type, COUNT(*) AS count
ORDER BY count DESC;

-- View all relationship types
MATCH ()-[r:RELATED_TO]->()
RETURN r.type AS relationship_type, COUNT(*) AS count
ORDER BY count DESC;

-- Find a specific entity and its neighbors
MATCH (e:Entity {normalized_name: "dopamine"})-[r]-(neighbor)
RETURN e, r, neighbor;

-- View community members
MATCH (e:Entity {community_id: 0})
RETURN e.name, e.entity_type, e.description
ORDER BY e.name;

-- Find path between entities
MATCH path = shortestPath(
  (a:Entity {normalized_name: "stress"})-[*..5]-(b:Entity {normalized_name: "prefrontal cortex"})
)
RETURN path;

-- Top 10 most connected entities (hubs)
MATCH (e:Entity)-[r]-()
RETURN e.name, e.entity_type, COUNT(r) AS connections
ORDER BY connections DESC
LIMIT 10;

-- Relationships within a community
MATCH (a:Entity {community_id: 0})-[r:RELATED_TO]->(b:Entity {community_id: 0})
RETURN a.name, r.type, b.name;

-- Graph statistics
MATCH (n) RETURN count(n) AS nodes;
MATCH ()-[r]->() RETURN count(r) AS relationships;
MATCH (e:Entity) RETURN count(DISTINCT e.community_id) AS communities;

-- Clear entire graph (WARNING: destructive)
MATCH (n) DETACH DELETE n;
```

### Visual Exploration Tips

```
1. Enter: MATCH (e:Entity)-[r]-(neighbor) RETURN e, r, neighbor LIMIT 100

2. Click "Graph" view (bubble icon in top-right)

3. Mouse controls:
   - Drag nodes to rearrange
   - Double-click to expand connections
   - Hover for properties
   - Scroll to zoom

4. Style by entity_type:
   - Click gear icon → "Graph Settings"
   - Set node color by: entity_type
   - Common colors:
     BRAIN_REGION = Blue
     NEUROTRANSMITTER = Green
     RESEARCHER = Orange
     PHILOSOPHER = Purple
```

---

## 12. Data Flow Trace

Complete trace from PDF to query response:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE DATA FLOW TRACE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1-4: DOCUMENT PROCESSING                                            │
│  ════════════════════════════════                                           │
│                                                                             │
│  data/raw/behave.pdf                                                        │
│       │                                                                     │
│       ▼ Stage 1: Docling extraction                                        │
│  data/processed/01_raw_extraction/behave.md                                 │
│       │                                                                     │
│       ▼ Stage 2: Regex cleaning                                            │
│  data/processed/03_markdown_cleaning/behave.md                              │
│       │                                                                     │
│       ▼ Stage 3: spaCy segmentation                                        │
│  data/processed/04_nlp_chunks/behave.json                                   │
│       │  [{"text": "The prefrontal cortex...", "sentence_id": 42}, ...]    │
│       │                                                                     │
│       ▼ Stage 4: 800-token chunking with 2-sentence overlap                │
│  data/processed/05_final_chunks/section/behave.json                         │
│       │  [{"text": "...", "chunk_id": "behave::chunk_42", ...}, ...]       │
│       │                                                                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 4.6: ENTITY EXTRACTION                                              │
│  ═══════════════════════════════                                            │
│                                                                             │
│  data/processed/05_final_chunks/section/behave.json                         │
│       │                                                                     │
│       ▼ For each chunk:                                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EXTRACTION PROMPT                                                    │   │
│  │ "Extract entities and relationships...                               │   │
│  │  Entity types: BRAIN_REGION, NEUROTRANSMITTER, ...                  │   │
│  │  Text: The prefrontal cortex regulates emotional responses..."       │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼ call_structured_completion()             │
│                                   model: anthropic/claude-3-haiku          │
│                                   response_model: ExtractionResult         │
│                                 │                                          │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │ LLM RESPONSE (JSON)                                                  │   │
│  │ {                                                                    │   │
│  │   "entities": [                                                      │   │
│  │     {"name": "prefrontal cortex", "entity_type": "BRAIN_REGION",    │   │
│  │      "description": "Brain region for executive function"}          │   │
│  │   ],                                                                 │   │
│  │   "relationships": [                                                 │   │
│  │     {"source_entity": "prefrontal cortex",                          │   │
│  │      "target_entity": "emotional responses",                        │   │
│  │      "relationship_type": "REGULATES", ...}                         │   │
│  │   ]                                                                  │   │
│  │ }                                                                    │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼ Add source_chunk_id to all items        │
│                                                                             │
│  data/processed/05_final_chunks/graph/extraction_results.json                            │
│  {                                                                          │
│    "entities": [450 entities with source tracking],                        │
│    "relationships": [320 relationships with source tracking],               │
│    "stats": {"total_entities": 450, "total_relationships": 320, ...}       │
│  }                                                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 6b: NEO4J UPLOAD + LEIDEN                                           │
│  ═════════════════════════════════                                          │
│                                                                             │
│  extraction_results.json                                                    │
│       │                                                                     │
│       ▼ upload_entities() with MERGE                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CYPHER: MERGE (e:Entity {normalized_name: "dopamine"})              │   │
│  │         ON CREATE SET e.name = "dopamine", ...                      │   │
│  │                                                                     │   │
│  │ Deduplication: "Dopamine" and "dopamine" merge to same node        │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│       │ upload_relationships() with MERGE                                  │
│       ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CYPHER: MATCH (source:Entity), (target:Entity)                      │   │
│  │         MERGE (source)-[r:RELATED_TO {type: $type}]->(target)       │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│       ▼ Neo4j graph now populated                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ NEO4J GRAPH                                                         │   │
│  │                                                                     │   │
│  │   [dopamine]──MODULATES──>[reward system]                           │   │
│  │       │                         │                                   │   │
│  │   INFLUENCES              PART_OF                                   │   │
│  │       │                         │                                   │   │
│  │       ▼                         ▼                                   │   │
│  │   [motivation]           [mesolimbic pathway]                       │   │
│  │                                                                     │   │
│  │   450 nodes, 320 relationships                                     │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│       ▼ project_graph() → GDS in-memory                                   │
│       ▼ run_leiden(resolution=1.0)                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LEIDEN OUTPUT                                                       │   │
│  │                                                                     │   │
│  │   Community 0: [dopamine, reward system, motivation, ...]           │   │
│  │   Community 1: [prefrontal cortex, decision-making, ...]            │   │
│  │   Community 2: [Marcus Aurelius, Stoicism, virtue, ...]             │   │
│  │   ...                                                               │   │
│  │                                                                     │   │
│  │   12 communities total                                              │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│       ▼ write_communities_to_neo4j()                                      │
│         (adds community_id property to each Entity node)                   │
│                                                                             │
│       ▼ summarize_community() for each                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LLM SUMMARY (for Community 0)                                       │   │
│  │                                                                     │   │
│  │ "This community centers on the neurobiology of reward and          │   │
│  │  motivation. Dopamine, a key neurotransmitter, modulates           │   │
│  │  the reward system and influences motivational states..."          │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│  data/processed/05_final_chunks/graph/communities.json                                   │
│  {                                                                          │
│    "communities": [12 communities with summaries],                         │
│    "total_count": 12,                                                       │
│    "total_members": 180                                                     │
│  }                                                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  QUERY TIME: HYBRID RETRIEVAL                                              │
│  ═════════════════════════════════                                          │
│                                                                             │
│  User Query: "How does dopamine affect decision-making?"                   │
│       │                                                                     │
│       ▼ extract_query_entities()                                           │
│         Regex + Neo4j lookup → ["dopamine", "decision-making"]             │
│                                                                             │
│       │                                                                     │
│       ├────────────────────────────────┬───────────────────────────────┐   │
│       │                                │                               │   │
│       ▼                                ▼                               ▼   │
│  ┌────────────┐               ┌─────────────────┐           ┌──────────┐   │
│  │ WEAVIATE   │               │     NEO4J       │           │COMMUNITY │   │
│  │ Vector     │               │   Graph Walk    │           │ MATCH    │   │
│  │ Search     │               │                 │           │          │   │
│  │            │               │ MATCH path =    │           │ Keyword  │   │
│  │ query_     │               │ (start)-[*1..2] │           │ overlap  │   │
│  │ hybrid()   │               │ -(neighbor)     │           │ scoring  │   │
│  │ alpha=0.5  │               │                 │           │          │   │
│  │            │               │ Collect:        │           │ Top 3    │   │
│  │ Returns:   │               │ source_chunk_id │           │ summaries│   │
│  │ [12,45,78] │               │ → [12,33,67]   │           │          │   │
│  └─────┬──────┘               └────────┬────────┘           └────┬─────┘   │
│        │                               │                         │         │
│        └───────────────────────────────┼─────────────────────────┘         │
│                                        │                                    │
│                                        ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ HYBRID MERGE                                                        │   │
│  │                                                                     │   │
│  │ Vector: [12, 45, 78]    Graph: [12, 33, 67]                        │   │
│  │                                                                     │   │
│  │ chunk_12 in BOTH → graph_boost = True                              │   │
│  │                                                                     │   │
│  │ Result: [chunk_12★, chunk_45, chunk_78, chunk_33, ...]             │   │
│  │                                                                     │   │
│  │ Metadata:                                                           │   │
│  │ - query_entities: ["dopamine", "decision-making"]                  │   │
│  │ - community_context: [3 summaries]                                  │   │
│  │ - boosted_count: 1                                                  │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ANSWER GENERATION                                                   │   │
│  │                                                                     │   │
│  │ Context:                                                            │   │
│  │ "## Relevant Themes (from document corpus)                         │   │
│  │  This community centers on the neurobiology of reward...           │   │
│  │                                                                     │   │
│  │  ## Related Concepts (from knowledge graph)                        │   │
│  │  - dopamine: Neurotransmitter involved in reward                   │   │
│  │  - prefrontal cortex: Executive function                           │   │
│  │                                                                     │   │
│  │  ## Retrieved Passages                                             │   │
│  │  [1] From Behave: Dopamine plays a crucial role in..."             │   │
│  │                                                                     │   │
│  │ → LLM generates comprehensive answer                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. RAPTOR vs GraphRAG Comparison

Both RAPTOR and GraphRAG are implemented in RAG1-Mini. They complement each other for different query types:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RAPTOR vs GraphRAG                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAPTOR                               GRAPHRAG                              │
│  (arXiv:2401.18059)                   (arXiv:2404.16130)                   │
│                                                                             │
│  ┌─────────────┐                      ┌─────────────────────────────┐      │
│  │   Book 1    │                      │      Knowledge Graph        │      │
│  │ Summary Tree│                      │                             │      │
│  │      │      │                      │   [A]─────[B]─────[C]       │      │
│  │   ┌──┼──┐   │                      │    │       │       │        │      │
│  │   │  │  │   │                      │   [D]─────[E]─────[F]       │      │
│  │  L0 L1 L2   │                      │    │              │        │      │
│  └─────────────┘                      │   [G]────────────[H]       │      │
│                                       │                             │      │
│  ┌─────────────┐                      │    Communities:             │      │
│  │   Book 2    │                      │    ┌───┐  ┌───┐  ┌───┐     │      │
│  │ Summary Tree│                      │    │ 1 │  │ 2 │  │ 3 │     │      │
│  │      │      │                      │    └───┘  └───┘  └───┘     │      │
│  │   ┌──┼──┐   │                      └─────────────────────────────┘      │
│  │  L0 L1 L2   │                                                           │
│  └─────────────┘                                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STRUCTURE                                                                  │
│  ─────────                                                                  │
│  RAPTOR:   Hierarchical tree (per-document)                                │
│  GraphRAG: Network graph (cross-document)                                  │
│                                                                             │
│  SCOPE                                                                      │
│  ─────                                                                      │
│  RAPTOR:   Within a single document                                        │
│  GraphRAG: Across all documents in corpus                                  │
│                                                                             │
│  BEST FOR                                                                   │
│  ────────                                                                   │
│  RAPTOR:   "Summarize chapter 3 of Behave"                                 │
│            "What does Sapolsky say about stress in this book?"             │
│            Deep questions within one document                              │
│                                                                             │
│  GraphRAG: "Compare views on virtue across all philosophy books"           │
│            "How do concepts from neuroscience relate to Stoicism?"         │
│            Cross-document synthesis and relationship queries               │
│                                                                             │
│  INDEX TIME                                                                 │
│  ──────────                                                                 │
│  RAPTOR:   Fast (GMM clustering + LLM summarization)                       │
│  GraphRAG: Slower (entity extraction per chunk + Leiden + summaries)       │
│                                                                             │
│  QUERY TIME                                                                 │
│  ──────────                                                                 │
│  RAPTOR:   Fast (collapsed tree search in Weaviate)                        │
│  GraphRAG: Medium (Neo4j traversal + vector search + merge)                │
│                                                                             │
│  STORAGE                                                                    │
│  ───────                                                                    │
│  RAPTOR:   Weaviate only (summaries as chunks)                             │
│  GraphRAG: Weaviate + Neo4j (dual-database)                                │
│                                                                             │
│  COMPLEXITY                                                                 │
│  ──────────                                                                 │
│  RAPTOR:   Medium (GMM, UMAP, summarization)                               │
│  GraphRAG: High (entity extraction, Neo4j, Leiden, community summaries)    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WHEN TO USE EACH                                                          │
│  ═══════════════                                                            │
│                                                                             │
│  Use RAPTOR when:                                                          │
│  ✓ Deep dive into a single document                                        │
│  ✓ Need hierarchical understanding (overview → details)                    │
│  ✓ Questions like "What are the key points of this chapter?"              │
│  ✓ Simpler infrastructure (Weaviate only)                                  │
│                                                                             │
│  Use GraphRAG when:                                                        │
│  ✓ Synthesizing across multiple documents                                  │
│  ✓ Relationship-based queries ("How does X relate to Y?")                 │
│  ✓ Entity-focused questions ("What does Sapolsky say about dopamine?")    │
│  ✓ Global queries ("What are the main themes in the corpus?")             │
│                                                                             │
│  Use BOTH together:                                                        │
│  ✓ RAPTOR for within-document hierarchy                                   │
│  ✓ GraphRAG for cross-document relationships                              │
│  ✓ Vector search as baseline for all queries                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PIPELINE COMMANDS                                                         │
│  ═════════════════                                                          │
│                                                                             │
│  RAPTOR:                                                                   │
│    python -m src.stages.run_stage_4_5_raptor                               │
│    (builds summary tree, stores in Weaviate)                               │
│                                                                             │
│  GraphRAG:                                                                 │
│    python -m src.stages.run_stage_4_6_graph_extract                        │
│    python -m src.stages.run_stage_6b_neo4j                                 │
│    (builds knowledge graph + communities in Neo4j)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. GraphRAG Research Updates (2025)

### Recent Developments from Microsoft Research

Based on the latest research (paper updated February 2025), here are key developments:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAPHRAG 2025 UPDATES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. AUTO-TUNING EXTRACTION PROMPTS                                         │
│  ══════════════════════════════════                                         │
│                                                                             │
│  Problem: Manual prompt engineering for entity types is time-consuming     │
│                                                                             │
│  Solution: Microsoft developed automatic prompt tuning that:               │
│  - Analyzes sample documents from the corpus                               │
│  - Identifies relevant entity/relationship types                           │
│  - Generates domain-specific extraction prompts                            │
│  - Reduces manual customization effort significantly                       │
│                                                                             │
│  Status: Available in Microsoft GraphRAG library                           │
│  RAG1-Mini: Uses manual domain types (BRAIN_REGION, PHILOSOPHER, etc.)     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2. NLP-BASED APPROXIMATION (LazyGraphRAG)                                 │
│  ═════════════════════════════════════════                                  │
│                                                                             │
│  Problem: LLM extraction is expensive ($1-10 per corpus)                   │
│                                                                             │
│  Solution: LazyGraphRAG uses NLP techniques to approximate:                │
│  - Named Entity Recognition (NER) instead of LLM extraction               │
│  - Dependency parsing for relationships                                    │
│  - Coreference resolution for entity linking                               │
│                                                                             │
│  Trade-off: Lower quality but 10-100x cheaper                              │
│  Use case: Quick prototyping or budget-constrained deployments             │
│                                                                             │
│  Status: Research preview                                                  │
│  RAG1-Mini: Uses full LLM extraction (learning project, quality focus)     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  3. MICROSOFT DISCOVERY INTEGRATION                                        │
│  ═══════════════════════════════════                                        │
│                                                                             │
│  Microsoft Discovery: Agentic platform for scientific research in Azure    │
│                                                                             │
│  GraphRAG integration provides:                                            │
│  - Managed Neo4j instances                                                 │
│  - Automated indexing pipelines                                            │
│  - Multi-tenant support                                                    │
│  - Enterprise security                                                     │
│                                                                             │
│  Relevance: Shows GraphRAG moving from research to production              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  4. KNOWN CHALLENGES (Being Addressed)                                     │
│  ═══════════════════════════════════════                                    │
│                                                                             │
│  Challenge 1: Entity Resolution                                            │
│  ─────────────────────────────────                                          │
│  Current: Match by normalized name only                                    │
│  Problem: "Marcus Aurelius" vs "Aurelius" treated as different             │
│  Future: Embedding-based similarity, coreference resolution               │
│                                                                             │
│  RAG1-Mini approach:                                                       │
│  - normalized_name for exact match (current)                               │
│  - Could add: embedding similarity for near-matches                        │
│                                                                             │
│  Challenge 2: Indexing Cost                                                │
│  ──────────────────────────                                                 │
│  Current: ~$1-5 per 1000 chunks (LLM extraction)                          │
│  Problem: Expensive for large corpora                                      │
│  Future: Hybrid NLP + LLM, incremental updates                            │
│                                                                             │
│  RAG1-Mini approach:                                                       │
│  - Claude-3-haiku ($0.25/1M tokens) for cost efficiency                   │
│  - ~150 chunks = ~$0.15 extraction cost                                   │
│                                                                             │
│  Challenge 3: Community Hierarchy Depth                                    │
│  ─────────────────────────────────────                                      │
│  Current: Single-level communities (level 0)                               │
│  Paper: Multi-level hierarchy (C0, C1, C2...)                             │
│  Benefit: Different abstraction levels for different queries              │
│                                                                             │
│  RAG1-Mini approach:                                                       │
│  - Uses single level (simpler implementation)                             │
│  - Could extend: store intermediate Leiden levels                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  5. EVALUATION ADVANCES                                                    │
│  ═══════════════════════                                                    │
│                                                                             │
│  New metrics from paper (Table 3):                                         │
│                                                                             │
│  Comprehensiveness: "How thoroughly does the answer cover the topic?"     │
│    - GraphRAG C0: 72-83% win rate vs baseline                             │
│                                                                             │
│  Diversity: "Does the answer explore multiple perspectives?"              │
│    - GraphRAG C0: 62-82% win rate vs baseline                             │
│                                                                             │
│  Empowerment: "Does the answer help the user take action?"                │
│    - GraphRAG C0: 35-48% win rate vs baseline (competitive)               │
│                                                                             │
│  Directness: "Does the answer directly address the question?"             │
│    - Baseline RAG wins here (GraphRAG adds too much context)              │
│                                                                             │
│  RAG1-Mini approach:                                                       │
│  - Uses RAGAS metrics (faithfulness, relevancy, context_precision)        │
│  - Could add: comprehensiveness, diversity via LLM-as-judge              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Paper Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1 | April 2024 | Original publication |
| v2 | February 2025 | Updated evaluation, LazyGraphRAG mention |

### Links to Latest Research

- [GraphRAG Paper v2 (Feb 2025)](https://arxiv.org/abs/2404.16130)
- [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [GraphRAG Auto-Tuning Blog](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/)
- [Microsoft Discovery Platform](https://www.microsoft.com/en-us/research/project/graphrag/)

---

## 15. Sources

### Primary Research
- [GraphRAG Paper (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130) - Microsoft Research, April 2024
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag/)

### Neo4j Resources
- [Neo4j Python Driver Manual](https://neo4j.com/docs/python-manual/current/)
- [Neo4j GDS Leiden Algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
- [Neo4j GraphRAG Python Package](https://neo4j.com/docs/neo4j-graphrag-python/current/)

### Algorithm References
- [Leiden Algorithm Wikipedia](https://en.wikipedia.org/wiki/Leiden_algorithm)
- [Modularity in Community Detection](https://en.wikipedia.org/wiki/Modularity_(networks))

### Implementation Files (this project)
- `src/graph/schemas.py` - Pydantic data models
- `src/graph/auto_tuning.py` - Auto-discover entity types from corpus
- `src/graph/extractor.py` - LLM entity extraction
- `src/graph/neo4j_client.py` - Neo4j operations
- `src/graph/community.py` - Leiden + summarization
- `src/graph/query.py` - Hybrid retrieval
- `src/stages/run_stage_4_5_autotune.py` - Auto-tuning stage (resumable)
- `src/stages/run_stage_4_6_graph_extract.py` - Extraction stage
- `src/stages/run_stage_6b_neo4j.py` - Neo4j upload stage
- `src/config.py` (lines 554-669) - GraphRAG configuration
