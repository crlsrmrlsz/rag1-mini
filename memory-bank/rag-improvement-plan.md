# RAG Improvement Plan: Comprehensive Quality Enhancement

## Overview

This plan implements three major RAG improvements (Contextual Embeddings, RAPTOR, GraphRAG) plus supporting infrastructure for UI-based testing and automatic RAGAS evaluation logging.

**Goal**: Improve answer quality through isolated, measurable experiments with easy A/B testing from the UI.

**Current Best Metrics** (Run 3):
- Relevancy: 0.786
- Faithfulness: 0.885
- Failures: 0/23 (0%)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
├─────────────────────────────────────────────────────────────────┤
│  Sidebar                    │  Main Area                        │
│  ┌───────────────────────┐  │  ┌─────────────────────────────┐ │
│  │ Collection Selector   │  │  │ Tabs: Answer | Pipeline |   │ │
│  │ ├─ RAG_section800_v1  │  │  │       Chunks               │ │
│  │ ├─ RAG_contextual_v1  │  │  └─────────────────────────────┘ │
│  │ ├─ RAG_raptor_v1      │  │                                   │
│  │ └─ RAG_graphrag_v1    │  │  (Evaluation runs via CLI:        │
│  │                       │  │   python -m src.run_stage_7_...)  │
│  │ Stage 1-4 Controls    │  │                                   │
│  │ (existing)            │  │                                   │
│  └───────────────────────┘  │                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE STAGES                            │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Chunking          │  Stage 5: Embedding               │
│  ├─ naive_chunker.py        │  ├─ embed_texts.py                │
│  ├─ contextual_chunker.py   │  └─ (same for all strategies)     │
│  ├─ raptor_chunker.py       │                                   │
│  └─ (run outside UI)        │  Stage 6: Weaviate                │
│                             │  ├─ weaviate_client.py            │
│                             │  └─ Multiple collections          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      GRAPHRAG (Neo4j)                           │
├─────────────────────────────────────────────────────────────────┤
│  Entity Extraction → Graph Construction → Graph + Vector Search │
│  ├─ src/graph/extractor.py                                      │
│  ├─ src/graph/neo4j_client.py                                   │
│  └─ src/graph/query.py                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: UI Foundation

**No UI changes needed!**

### Collection Selector

**Already implemented** at `src/ui/app.py:220-235`:
- Lists all `RAG_*` collections from Weaviate via `list_collections()`
- Dropdown at top of sidebar
- Selected collection passed to `search_chunks()` via `collection_name` parameter

New collections (contextual, raptor, graphrag) will automatically appear in the dropdown after running Stage 6.

### Evaluation (CLI Only)

Evaluation stays outside the UI for simplicity. Improvements needed for `src/run_stage_7_evaluation.py`:

**Current CLI arguments (already implemented):**
```bash
python -m src.run_stage_7_evaluation [OPTIONS]

Options:
  -n, --questions N         Limit to first N questions
  -c, --category CATEGORY   Filter: neuroscience|philosophy|synthesis|open_ended
  -m, --metrics METRICS     Metrics to compute (default: faithfulness relevancy context_precision)
  -k, --top-k K             Chunks to retrieve (default: 10)
  --generation-model MODEL  Answer generation model (default: openai/gpt-5-mini)
  --evaluation-model MODEL  RAGAS judge model (default: anthropic/claude-haiku-4.5)
  -a, --alpha ALPHA         Hybrid search: 0.0=keyword, 0.5=balanced, 1.0=vector
  --reranking/--no-reranking  Enable/disable cross-encoder
  -o, --output PATH         Output file path
```

**To add:**
1. `--collection` argument to select which Weaviate collection (e.g., `RAG_contextual_v1`)
2. Auto-append results to `memory-bank/evaluation-history.md`
3. Update `data/evaluation/tracking.json` with run config

**Implementation:**

```python
# Add to argparse:
parser.add_argument(
    "--collection",
    type=str,
    default=None,
    help="Weaviate collection to evaluate (default: auto from config)",
)

# Add auto-logging function:
def append_to_evaluation_history(results, config, output_path):
    """Append run summary to memory-bank/evaluation-history.md"""
    history_path = Path("memory-bank/evaluation-history.md")

    # Get next run number
    run_number = get_next_run_number(history_path)

    # Format markdown
    entry = f"""
---

## Run {run_number}: {config['collection']}

**Date:** {datetime.now().strftime('%B %d, %Y')}
**File:** `{output_path.relative_to(Path.cwd())}`

### Configuration
- **Collection:** {config['collection']}
- **Search Type:** Hybrid
- **Alpha:** {config['alpha']}
- **Top-K:** {config['top_k']}
- **Reranking:** {'Yes' if config['reranking'] else 'No'}
- **Generation Model:** {config['generation_model']}
- **Evaluation Model:** {config['evaluation_model']}

### Results
| Metric | Score |
|--------|-------|
| Faithfulness | {results['scores'].get('faithfulness', 'N/A'):.3f} |
| Relevancy | {results['scores'].get('relevancy', 'N/A'):.3f} |
| Context Precision | {results['scores'].get('context_precision', 'N/A'):.3f} |

### Key Learning
[Add notes about this run manually]
"""

    with open(history_path, "a") as f:
        f.write(entry)

    logger.info(f"Appended to evaluation-history.md as Run {run_number}")
```

**Example usage after improvements:**
```bash
# Test contextual embeddings collection
python -m src.run_stage_7_evaluation --collection RAG_contextual_embed3large_v1

# Test RAPTOR with different alpha
python -m src.run_stage_7_evaluation --collection RAG_raptor_embed3large_v1 --alpha 0.7

# Compare collections
python -m src.run_stage_7_evaluation --collection RAG_section800_embed3large_v1
python -m src.run_stage_7_evaluation --collection RAG_contextual_embed3large_v1
```

---

## Phase 1: Contextual Retrieval (Anthropic-style)

**Expected Impact**: +35% reduction in retrieval failures (per Anthropic research)

**Concept**: Before embedding each chunk, prepend an LLM-generated context summary that describes where the chunk fits in the document.

### 1.1 Create Contextual Chunker

**New file**: `src/ingest/contextual_chunker.py`

```python
def create_contextual_chunks(chunks: List[Dict], book_text: str) -> List[Dict]:
    """Add contextual prefix to each chunk before embedding."""
    for chunk in chunks:
        context_prompt = f"""
Document: {chunk['book_id']}
Section: {chunk['context']}

Here is the chunk:
{chunk['text']}

Provide a brief (2-3 sentence) context that situates this chunk
within the document. Start with "This passage..."
"""
        context = call_openrouter_chat(context_prompt, model=PREPROCESSING_MODEL)
        chunk['contextual_text'] = f"{context}\n\n{chunk['text']}"
        chunk['context_summary'] = context
    return chunks
```

### 1.2 New Stage 4 Variant

**New file**: `src/run_stage_4_contextual.py`

1. Run standard chunking
2. For each chunk, generate context summary via LLM
3. Prepend context to chunk text
4. Save to `data/processed/05_final_chunks/contextual/`

### 1.3 Embedding & Upload

- Run `python -m src.run_stage_5_embedding` (reads from contextual/)
- Run `python -m src.run_stage_6_weaviate` with `CHUNKING_STRATEGY_NAME=contextual`
- Creates collection: `RAG_contextual_embed3large_v1`

### 1.4 Test from UI

1. Select `RAG_contextual_embed3large_v1` in collection dropdown
2. Run same queries as baseline
3. Trigger RAGAS evaluation
4. Compare metrics to Run 3 baseline

---

## Phase 2: RAPTOR (Hierarchical Summarization)

**Expected Impact**: +20% improvement on long-document comprehension

**Concept**: Build a tree where leaves are chunks, nodes are LLM summaries of clustered chunks. Retrieval can match at any level.

### 2.1 RAPTOR Architecture

```
Level 3:  [Book Summary]
              │
Level 2:  [Theme A] ─── [Theme B] ─── [Theme C]
              │              │             │
Level 1:  [Sec1][Sec2]   [Sec3][Sec4]  [Sec5][Sec6]
              │              │             │
Level 0:  [c1][c2][c3]   [c4][c5]      [c6][c7][c8]  ← Original chunks
```

### 2.2 Create RAPTOR Chunker

**New file**: `src/ingest/raptor_chunker.py`

```python
def build_raptor_tree(chunks: List[Dict], max_levels: int = 3) -> List[Dict]:
    """Build hierarchical summary tree from chunks."""
    all_nodes = []
    current_level = chunks.copy()

    for level in range(max_levels):
        # 1. Embed current level
        embeddings = embed_texts([c['text'] for c in current_level])

        # 2. Cluster by semantic similarity (k-means or HDBSCAN)
        clusters = cluster_embeddings(embeddings, target_clusters=len(current_level)//3)

        # 3. Generate summary for each cluster
        summaries = []
        for cluster_chunks in clusters:
            combined_text = "\n\n".join([c['text'] for c in cluster_chunks])
            summary = generate_cluster_summary(combined_text)
            summaries.append({
                'text': summary,
                'level': level + 1,
                'children': [c['chunk_id'] for c in cluster_chunks],
                'chunk_id': f"raptor_L{level+1}_{uuid4().hex[:8]}"
            })

        all_nodes.extend(current_level)
        current_level = summaries

        if len(current_level) <= 1:
            break

    all_nodes.extend(current_level)  # Add top-level summaries
    return all_nodes
```

### 2.3 RAPTOR Collection Schema

Weaviate schema additions:
- `raptor_level`: int (0=leaf, 1+=summary)
- `children`: text[] (child chunk_ids)
- `parent`: text (parent chunk_id)

### 2.4 RAPTOR Query Strategy

```python
def query_raptor(query: str, collection: str, top_k: int = 10):
    # 1. Search across all levels
    results = query_hybrid(query, top_k=top_k*2, collection=collection)

    # 2. For each summary node, optionally fetch children
    expanded = []
    for r in results:
        if r.raptor_level > 0 and r.score > 0.7:
            # High-scoring summary: fetch children for detail
            children = fetch_children(r.children)
            expanded.extend(children)
        else:
            expanded.append(r)

    # 3. Deduplicate and re-rank
    return deduplicate_and_rank(expanded)[:top_k]
```

### 2.5 New Stage 4 Variant

**New file**: `src/run_stage_4_raptor.py`

1. Load standard chunks
2. Build RAPTOR tree (3 levels)
3. Save all nodes (leaves + summaries) to `data/processed/05_final_chunks/raptor/`

### 2.6 Test from UI

1. Select `RAG_raptor_embed3large_v1`
2. Test queries requiring synthesis (cross-domain questions)
3. Run RAGAS evaluation
4. Compare to baseline and contextual

---

## Phase 3: GraphRAG (Neo4j Integration)

**Expected Impact**: +70-80% win rate on comprehensiveness vs baseline (Microsoft research)

**Concept**: Extract entities and relationships, build knowledge graph in Neo4j, combine graph traversal with vector search.

### 3.1 Neo4j Setup

```bash
# Docker Compose addition
services:
  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data
```

### 3.2 Entity Extraction

**New file**: `src/graph/extractor.py`

```python
def extract_entities_and_relations(chunk: Dict) -> Dict:
    """Extract entities and relationships using LLM."""
    prompt = f"""
Extract entities and relationships from this text.

Text: {chunk['text']}

Return JSON:
{{
  "entities": [
    {{"name": "...", "type": "CONCEPT|PERSON|THEORY|BRAIN_REGION|..."}}
  ],
  "relationships": [
    {{"source": "...", "relation": "DEFINES|ARGUES|RELATES_TO|...", "target": "..."}}
  ]
}}
"""
    return call_openrouter_json(prompt, model=PREPROCESSING_MODEL)
```

### 3.3 Neo4j Client

**New file**: `src/graph/neo4j_client.py`

```python
from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_entity(self, name: str, entity_type: str, chunk_ids: List[str]):
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $type, e.chunk_ids = $chunk_ids
        RETURN e
        """
        # Execute query

    def create_relationship(self, source: str, relation: str, target: str):
        query = """
        MATCH (a:Entity {name: $source})
        MATCH (b:Entity {name: $target})
        MERGE (a)-[r:RELATES {type: $relation}]->(b)
        RETURN r
        """
        # Execute query

    def query_neighborhood(self, entity: str, hops: int = 2) -> List[str]:
        """Get chunk_ids from entity neighborhood."""
        query = f"""
        MATCH (e:Entity {{name: $entity}})-[*1..{hops}]-(related)
        RETURN DISTINCT related.chunk_ids as chunks
        """
        # Return merged chunk_ids
```

### 3.4 Graph-Augmented Query

**New file**: `src/graph/query.py`

```python
def query_graphrag(query: str, top_k: int = 10) -> List[SearchResult]:
    """Combine graph traversal with vector search."""
    # 1. Extract entities from query
    query_entities = extract_entities_from_query(query)

    # 2. Get chunk_ids from graph neighborhood
    graph_chunks = set()
    for entity in query_entities:
        neighborhood = neo4j_client.query_neighborhood(entity, hops=2)
        graph_chunks.update(neighborhood)

    # 3. Vector search for semantic matches
    vector_results = query_hybrid(query, top_k=top_k*2)

    # 4. Boost chunks found in graph
    for result in vector_results:
        if result.chunk_id in graph_chunks:
            result.score *= 1.3  # Graph boost

    # 5. Re-rank and return
    return sorted(vector_results, key=lambda x: x.score, reverse=True)[:top_k]
```

### 3.5 GraphRAG Pipeline Stages

**New files**:
- `src/run_stage_4b_graph_extract.py` - Extract entities/relations from chunks
- `src/run_stage_6b_neo4j.py` - Upload graph to Neo4j

### 3.6 UI Integration

Add search mode toggle:
- Vector Only (existing)
- Hybrid (existing)
- GraphRAG (new) - Uses `query_graphrag()`

---

## Phase 4: Step-Back Prompt Enhancement

**Expected Impact**: Improved retrieval precision through domain-specific vocabulary

**Concept**: Enhance the step-back prompting technique with Chain-of-Thought examples and concrete vocabulary from the knowledge base.

**Research**: See `memory-bank/step-back-prompting-research.md` for full analysis.

### 4.1 Current Problem

The current step-back prompt generates generic queries like:
```
"neuroscience of social validation; philosophical ethics of external approval"
```

This is too abstract and doesn't use vocabulary that matches chunk content.

### 4.2 Improved Prompt

Replace `STEP_BACK_PROMPT` in `src/preprocessing/query_classifier.py` with a version that:
1. Uses Chain-of-Thought examples ("Think: Core=..., Mechanisms=...")
2. Includes domain-specific vocabulary (author names, brain regions, philosophical schools)
3. Generates mixed-term queries for better hybrid retrieval

**Expected output after improvement**:
```
"dopamine social reward approval seeking Stoic virtue external validation Marcus Aurelius"
```

### 4.3 Implementation

**File**: `src/preprocessing/query_classifier.py` (lines 268-286)
**Change**: Prompt text only, no code logic changes

### 4.4 Future Iterations (if metrics improve)

1. **Multi-Query Generation**: Generate 4 targeted queries + RRF merging
2. **Principle Extraction**: Add explicit concept extraction step

---

## Phase 5: Query Decomposition (MULTI_HOP)

**Expected Impact**: +36.7% MRR@10 improvement

**Concept**: Break complex queries into sub-questions, retrieve for each, merge results.

### 5.1 Implement Decomposition

**Modify**: `src/preprocessing/query_classifier.py`

```python
def decompose_query(query: str, model: Optional[str] = None) -> List[str]:
    """Decompose multi-hop query into sub-questions."""
    prompt = f"""
Break this complex question into 2-4 simpler sub-questions that can be answered independently:

Question: {query}

Return JSON array of sub-questions:
["sub-question 1", "sub-question 2", ...]
"""
    result = call_openrouter_json(prompt, model=model or PREPROCESSING_MODEL)
    return result  # List[str]
```

### 5.2 Multi-Query Retrieval

```python
def retrieve_multi_hop(query: str, sub_queries: List[str], top_k: int) -> List[SearchResult]:
    """Retrieve for each sub-query, merge with RRF."""
    all_results = {}

    for sub_q in sub_queries:
        results = query_hybrid(sub_q, top_k=top_k)
        for rank, r in enumerate(results):
            if r.chunk_id not in all_results:
                all_results[r.chunk_id] = {'result': r, 'rrf_score': 0}
            all_results[r.chunk_id]['rrf_score'] += 1 / (60 + rank)  # RRF formula

    # Sort by RRF score
    merged = sorted(all_results.values(), key=lambda x: x['rrf_score'], reverse=True)
    return [m['result'] for m in merged][:top_k]
```

### 5.3 Update Preprocessing Flow

```python
elif query_type == QueryType.MULTI_HOP:
    sub_queries = decompose_query(query, model=model)
    search_query = query  # Keep original for display
    # Store sub_queries in PreprocessedQuery for retrieval stage
```

---

## Phase 6: Quick Wins (High Impact, Low Effort)

### 6.1 Lost-in-the-Middle Mitigation

**Impact**: +15% answer quality

**Modify**: `src/generation/answer_generator.py`

```python
def reorder_chunks_for_attention(chunks: List[Dict]) -> List[Dict]:
    """Place best chunks at start and end, weaker in middle."""
    if len(chunks) <= 3:
        return chunks

    # Assume chunks are ranked by relevance
    best = chunks[:2]
    worst = chunks[2:-2]
    second_best = chunks[-2:]

    return best + worst + second_best
```

### 6.2 Alpha Tuning Experiments

Already in task list. Run via CLI:

```bash
python -m src.run_stage_7_evaluation --alpha 0.3  # Keyword-heavy (philosophy)
python -m src.run_stage_7_evaluation --alpha 0.5  # Balanced (default)
python -m src.run_stage_7_evaluation --alpha 0.7  # Vector-heavy (conceptual)
```

Update `evaluation-history.md` with results after each run.

---

## Implementation Order

| Order | Phase | Effort | Impact | Files |
|-------|-------|--------|--------|-------|
| 0 | Evaluation CLI improvements | Low | Enables A/B testing | `run_stage_7_evaluation.py` |
| 1 | Phase 4: Step-Back Prompt Enhancement | Low | Better retrieval | `query_classifier.py` (prompt only) |
| 2 | Phase 6.1: Lost-in-middle | Low | +15% | `answer_generator.py` |
| 3 | Phase 1: Contextual | Medium | +35% failures | `contextual_chunker.py`, Stage 4 |
| 4 | Phase 5: MULTI_HOP | Low | +36.7% MRR | `query_classifier.py` |
| 5 | Phase 2: RAPTOR | High | +20% comprehension | `raptor_chunker.py`, Stage 4 |
| 6 | Phase 3: GraphRAG | High | +70% coverage | `graph/`, Neo4j |

**Note**: Phase 4 (Step-Back Prompt Enhancement) research and implementation details are in `memory-bank/step-back-prompting-research.md`.

---

## Testing Protocol

For each improvement:

1. **Create new collection** (run Stage 4-6 via CLI)
2. **Select collection** in UI dropdown
3. **Run manual queries** in UI to verify basic functionality
4. **Run RAGAS evaluation** via CLI: `python -m src.run_stage_7_evaluation`
5. **Update evaluation-history.md** with results
6. **Compare to baseline** (Run 3: 0.786 relevancy, 0.885 faithfulness)

---

## Files to Create/Modify

### New Files
- `src/ingest/contextual_chunker.py` - Contextual enrichment
- `src/ingest/raptor_chunker.py` - RAPTOR tree building
- `src/run_stage_4_contextual.py` - Contextual chunking stage
- `src/run_stage_4_raptor.py` - RAPTOR chunking stage
- `src/graph/extractor.py` - Entity extraction
- `src/graph/neo4j_client.py` - Neo4j connection
- `src/graph/query.py` - Graph-augmented retrieval
- `src/run_stage_4b_graph_extract.py` - Graph extraction stage
- `src/run_stage_6b_neo4j.py` - Neo4j upload stage

### Modified Files
- `src/run_stage_7_evaluation.py` - Add --collection arg, auto-logging to evaluation-history.md
- `src/config.py` - Add graph config
- `src/preprocessing/query_classifier.py` - Improve STEP_BACK_PROMPT (Phase 4), add query decomposition (Phase 5)
- `src/generation/answer_generator.py` - Lost-in-middle fix
- `src/vector_db/weaviate_client.py` - RAPTOR schema fields
- `docker-compose.yml` - Add Neo4j service

### Research Files
- `memory-bank/step-back-prompting-research.md` - Full analysis of step-back technique with improved prompts

---

## Success Criteria

| Metric | Baseline (Run 3) | Target |
|--------|------------------|--------|
| Relevancy | 0.786 | > 0.85 |
| Faithfulness | 0.885 | > 0.92 |
| Failures | 0/23 | 0/23 |
| MULTI_HOP queries | Not handled | Decomposed & merged |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM cost for contextual enrichment | Use gpt-5-nano, batch processing |
| RAPTOR clustering quality | Test with different k values, use HDBSCAN |
| Neo4j complexity | Start with simple entity types, expand gradually |
| Over-engineering | Implement in order, stop when targets met |
