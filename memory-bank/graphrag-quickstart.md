# GraphRAG Quick Start

One-page reference for running GraphRAG on your corpus.

---

## Prerequisites

```bash
conda activate rag1-mini
docker compose up -d neo4j    # Start Neo4j (port 7474 for browser, 7687 for bolt)
```

**Verify Neo4j is running:** http://localhost:7474 (login: neo4j / rag1mini_graphrag)

---

## Execution Steps

### Step 1: Run Entity Extraction

Choose ONE of these options:

```bash
# OPTION A: Auto-Tuning (Recommended)
# Discovers entity types FROM your corpus
python -m src.stages.run_stage_4_5_autotune --strategy section

# OPTION B: Predefined Types
# Uses entity types hardcoded in src/config.py
python -m src.stages.run_stage_4_6_graph_extract --strategy section
```

**Output:** `data/processed/07_graph/extraction_results.json`

### Step 2: Upload to Neo4j + Run Leiden

```bash
python -m src.stages.run_stage_6b_neo4j
```

**What this does:**
1. Uploads entities and relationships to Neo4j
2. Runs Leiden community detection algorithm
3. Generates LLM summaries for each community

**Output:** `data/processed/07_graph/communities.json`

### Step 3: Query with GraphRAG

**Via UI:**
- Select "graphrag" in the preprocessing strategy dropdown

**Via CLI:**
```bash
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag
```

---

## Data Flow Diagram

```
┌──────────────┐     ┌─────────────────────────┐     ┌─────────────────────┐
│   Stage 4    │     │  Stage 4.5 autotune     │     │     Stage 6b        │
│   Chunking   │────▶│       OR                │────▶│   Neo4j + Leiden    │
│              │     │  Stage 4.6 extract      │     │                     │
└──────────────┘     └─────────────────────────┘     └─────────────────────┘
       │                        │                              │
       ▼                        ▼                              ▼
05_final_chunks/         07_graph/                      07_graph/
section/*.json       extraction_results.json         communities.json
                                                            +
                                                    Neo4j knowledge graph
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| Chunks | `data/processed/05_final_chunks/section/` | Input for extraction |
| Entities | `data/processed/07_graph/extraction_results.json` | Entities + relationships |
| Types | `data/processed/07_graph/discovered_types.json` | Auto-discovered types (only from 4.5) |
| Communities | `data/processed/07_graph/communities.json` | Leiden clusters + summaries |

---

## Useful Neo4j Queries

**Open browser:** http://localhost:7474

```cypher
// Count all entities
MATCH (e:Entity) RETURN count(e) as entity_count

// View entity types distribution
MATCH (e:Entity) RETURN e.entity_type, count(*) as count ORDER BY count DESC

// View a specific community
MATCH (e:Entity {community_id: 0}) RETURN e

// Find relationships for an entity
MATCH (e:Entity {normalized_name: 'dopamine'})-[r]-(n) RETURN e, r, n LIMIT 20

// View all communities with member counts
MATCH (e:Entity)
WHERE e.community_id IS NOT NULL
RETURN e.community_id, count(*) as size
ORDER BY size DESC
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Neo4j connection refused | Run `docker compose up -d neo4j` and wait 30s |
| "No extraction results found" | Run Stage 4.5 or 4.6 first |
| Leiden fails with "GDS not found" | Ensure Neo4j GDS plugin is installed (check docker-compose.yml) |
| Empty communities | Increase `GRAPHRAG_MIN_COMMUNITY_SIZE` in config.py |

---

## Key Configuration (src/config.py)

```python
# Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "rag1mini_graphrag"

# Leiden algorithm
GRAPHRAG_LEIDEN_RESOLUTION = 1.0    # Higher = more smaller communities
GRAPHRAG_MIN_COMMUNITY_SIZE = 3     # Skip tiny communities

# Query-time retrieval
GRAPHRAG_TOP_COMMUNITIES = 3        # Communities to include in context
GRAPHRAG_TRAVERSE_DEPTH = 2         # Hops from query entities
```

---

## See Also

- `memory-bank/graphrag-tutorial.md` - Full technical deep-dive
- `memory-bank/graphrag-research.md` - Research notes and paper analysis
- [Microsoft GraphRAG Paper (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130)
