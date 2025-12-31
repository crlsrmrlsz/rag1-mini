# GraphRAG: Knowledge Graph + Communities

**Date:** 2025-12-29
**Status:** All Phases Complete
**Paper:** [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) (Microsoft Research, April 2024)

> **Related Docs:**
> - [Public documentation](../docs/preprocessing/graphrag.md) — User-facing implementation guide with crash-proof design
> - [2025 SOTA Report](graphrag-sota-report.md) — Implementation landscape and benchmarks
> - [Deep Tutorial](graphrag-tutorial.md) — Comprehensive step-by-step guide

---

## Quick Reference

### Prerequisites

```bash
conda activate raglab
docker compose up -d neo4j weaviate
```

**Verify services:**
- Neo4j: http://localhost:7474 (neo4j / raglab_graphrag)
- Weaviate: http://localhost:8080

### Execution Steps

```bash
# Step 1: Entity Extraction (choose ONE)
python -m src.stages.run_stage_4_5_autotune --strategy section    # Auto-discovers types (recommended)
python -m src.stages.run_stage_4_6_graph_extract --strategy section  # Uses predefined types

# Step 1b: Re-consolidate for mixed corpora (optional)
python -m src.stages.run_stage_4_5_autotune --reconsolidate stratified

# Step 2: Upload + Leiden + Summarization
python -m src.stages.run_stage_6b_neo4j

# Step 3: Query
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag
```

### Crash Recovery

Stage 6b is crash-proof. Resume anytime:

```bash
# Resume after crash (checks Weaviate for existing communities)
python -m src.stages.run_stage_6b_neo4j --resume

# Full re-run after Neo4j reset (deterministic Leiden guarantees same IDs)
python -m src.stages.run_stage_6b_neo4j
```

**How it works:**
- Deterministic Leiden: `randomSeed=42` + `concurrency=1` ensures same community IDs
- Weaviate storage: Community embeddings in Weaviate (~12MB vs 383MB JSON)
- Atomic uploads: Each community saved immediately, resume skips existing

See `docs/preprocessing/graphrag.md#crash-proof-design-v2` for full details.

### Data Flow

```
Stage 4 → Stage 4.5/4.6 → Stage 6b → Query
Chunks    extraction_results.json    Neo4j + Weaviate
          discovered_types.json      communities.json (backup)
                                     leiden_checkpoint.json
```

### Useful Neo4j Queries

```cypher
-- Count entities
MATCH (e:Entity) RETURN count(e)

-- Entity types distribution
MATCH (e:Entity) RETURN e.entity_type, count(*) as count ORDER BY count DESC

-- Find entity relationships
MATCH (e:Entity {normalized_name: 'dopamine'})-[r]-(n) RETURN e, r, n LIMIT 20

-- Community sizes
MATCH (e:Entity) WHERE e.community_id IS NOT NULL
RETURN e.community_id, count(*) as size ORDER BY size DESC
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Neo4j connection refused | `docker compose up -d neo4j`, wait 30s |
| "No extraction results" | Run Stage 4.5 or 4.6 first |
| Leiden "GDS not found" | Check `NEO4J_PLUGINS` in docker-compose.yml |
| Empty communities | Increase `GRAPHRAG_MIN_COMMUNITY_SIZE` |

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Enhanced Entity Resolution | COMPLETE |
| Phase 2 | Community Embedding Retrieval | COMPLETE |
| Phase 3 | Verification and Documentation | COMPLETE |
| Phase 4 | Crash-Proof Design + Weaviate Storage | COMPLETE |

**Tests:** 30 passing (`pytest tests/ -v`)

### Key Improvements Implemented

1. **Entity Resolution** (Phase 1)
   - Unicode NFKC normalization
   - Edge stopword removal (the, a, an, of, in, on, for, to, and)
   - Punctuation stripping
   - Python pre-computation before Neo4j upload

2. **Community Embedding Retrieval** (Phase 2)
   - Summaries generate embeddings during summarization
   - Cosine similarity replaces keyword matching
   - Falls back to keyword if embeddings unavailable

3. **Crash-Proof Design** (Phase 4)
   - Deterministic Leiden with fixed seed
   - Weaviate storage for community embeddings
   - Checkpoint file for Leiden assignments
   - Storage reduced from 383MB to ~12MB

---

## Core Theory

### The Problem

Vector search fails on "global" questions:
```
Query: "What are the main themes across all 19 books?"
```

No single chunk contains this answer. GraphRAG solves this with knowledge graphs + community summaries.

### Key Results (Paper)

- **72-83% win rate** on comprehensiveness vs baseline RAG
- **62-82% win rate** on diversity of answers
- **97% fewer tokens** at query time using community summaries

### Two-Phase Architecture

```
INDEXING (Offline)
  Text Chunks → Entity Extraction → Knowledge Graph → Leiden Communities → LLM Summaries

QUERY (Online)
  User Query → Extract Entities → Graph Traversal + Vector Search → RRF Merge → Answer
```

### Key Terminology

- **Entity**: Named concept (person, brain region, philosophy)
- **Relationship**: Typed connection between entities
- **Community**: Cluster of related entities (Leiden algorithm)
- **Community Summary**: LLM-generated thematic description

### Entity Types (Domain-Specific)

```python
ENTITY_TYPES = [
    # Neuroscience
    "BRAIN_REGION", "NEUROTRANSMITTER", "COGNITIVE_PROCESS", "RESEARCHER",
    # Philosophy
    "PHILOSOPHER", "PHILOSOPHICAL_SCHOOL", "CONCEPT", "PRACTICE",
    # Cross-domain
    "BOOK", "THEORY", "PRINCIPLE"
]
```

### Leiden Algorithm

Hierarchical community detection:
1. Local moving (nodes join communities)
2. Refinement (verify connectivity)
3. Aggregation (recursive hierarchy)

**RAGLab config:**
- `randomSeed=42`, `concurrency=1` for determinism
- `gamma=1.0`, `maxLevels=10`
- Level 0 used for all queries

### Neo4j Schema

```cypher
(:Entity {
  name, normalized_name, entity_type, description,
  chunk_ids, mention_count, community_id
})

(:Entity)-[:RELATED_TO {type, description, strength, chunk_ids}]->(:Entity)
```

### Hybrid Retrieval (Query Time)

1. Extract entities from query (LLM + regex fallback)
2. Graph traversal from entities (N-hop neighbors)
3. Vector search in Weaviate
4. RRF merge (boost graph-matched chunks)
5. Add community summaries to context
6. Generate answer

---

## Implementation Analysis

### What Works Well

- Core Leiden pipeline correctly implemented
- Neo4j integration with proper MERGE patterns
- Auto-tuning with stratified consolidation (not in original paper)
- Query entity extraction with LLM + fallback chain
- Crash recovery with `--resume`
- Enhanced entity resolution (Unicode, stopwords, punctuation)
- Community embedding retrieval (cosine similarity)

### Deviations from Original Paper

| Feature | Original | This Project | Reason |
|---------|----------|--------------|--------|
| Map-reduce global search | Parallel partial answers | Embedding similarity | Simpler, captures essence |
| Community hierarchy | Multi-level (C0-Cn) | Level 0 only | Hard to measure benefit |
| Self-reflection loop | 3 iterations | Single pass | 3x cost for ~20% gain |
| Claims extraction | Verifiable facts | Not implemented | Scope reduction |

### Configuration Comparison

| Parameter | Original Paper | This Project |
|-----------|---------------|--------------|
| Chunk size | 600 tokens | 800 tokens (section-aware) |
| Leiden resolution | 1.0 | 1.0 |
| Max hierarchy levels | 10 | 10 |
| Min community size | Not specified | 3 |

---

## Related

- `graphrag-tutorial.md` - Deep technical tutorial (25k+ lines)
- `graphrag-sota-report.md` - 2025 research landscape
- `docs/preprocessing/graphrag.md` - User documentation with crash-proof design details

---

*Last Updated: 2025-12-29*
