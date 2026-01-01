# Preprocessing Strategies

Preprocessing transforms queries before retrieval. This is a **query-time decision** — you can switch strategies without re-indexing.

## Search Type vs Preprocessing (Important Distinction)

RAG retrieval has two orthogonal configuration axes:

| Axis | What It Controls | Options |
|------|-----------------|---------|
| **Search Type** | HOW Weaviate searches | `keyword` (BM25 only), `hybrid` (vector + BM25) |
| **Preprocessing** | Query transformation BEFORE search | `none`, `hyde`, `decomposition`, `graphrag` |

Any preprocessing strategy works with any search type. For example:
- `hyde` + `keyword`: Generate hypothetical answer, then BM25 search
- `decomposition` + `hybrid`: Sub-questions with vector+keyword search each

## Strategy Comparison

| Strategy | LLM Calls | Latency | Best For |
|----------|-----------|---------|----------|
| None | 0 | ~0ms | Baseline, simple factual queries |
| [HyDE](hyde.md) | 1 | ~500ms | Vague queries, semantic matching |
| [Decomposition](query-decomposition.md) | 1 | ~500ms | Complex multi-part questions |
| [GraphRAG](graphrag.md) | 1+ | ~1-2s | Cross-document synthesis |

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PREPROCESSING STRATEGY (query transformation)              │
│                                                             │
│  none:          query → query (unchanged)                   │
│  hyde:          query → hypothetical_passage(s)             │
│  decomposition: query → [sub_query_1, sub_query_2, ...]     │
│  graphrag:      query → query + entity_hints                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  SEARCH TYPE (Weaviate query method)                        │
│                                                             │
│  keyword:  BM25 only (pure term matching)                   │
│  hybrid:   BM25 + Vector (alpha controls balance)           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL                                                  │
│                                                             │
│  Single query:  Search with configured search_type          │
│  Multi-query:   Search each → RRF merge                     │
│  GraphRAG:      Search + Neo4j traversal → RRF merge        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Ranked Chunks → Generation
```

## Strategy Registry Pattern

All strategies implement the same interface:

```python
# src/rag_pipeline/retrieval/preprocessing/strategies.py

StrategyFunction = Callable[[str, Optional[str]], PreprocessedQuery]

STRATEGIES: Dict[str, StrategyFunction] = {
    "none": none_strategy,
    "hyde": hyde_strategy,
    "decomposition": decomposition_strategy,
    "graphrag": graphrag_strategy,
}

def get_strategy(strategy_id: str) -> StrategyFunction:
    """Get strategy function by ID."""
    return STRATEGIES[strategy_id]
```

This enables:
- A/B testing via CLI args or UI dropdown
- Easy addition of new strategies
- Consistent evaluation across strategies

## RRF Merging

When a strategy produces multiple search queries (decomposition, graphrag), results are merged using **Reciprocal Rank Fusion**:

```
RRF_score(doc) = sum(1 / (k + rank(doc, query_i))) for each query_i
```

Key properties:
- No score normalization needed
- Documents in multiple result lists get boosted
- `k=60` (standard value) controls rank influence decay

See `src/rag_pipeline/retrieval/rrf.py` for implementation.

## Running Strategies

```bash
# Via CLI - preprocessing strategies with default hybrid search
python -m src.stages.run_stage_7_evaluation --preprocessing none
python -m src.stages.run_stage_7_evaluation --preprocessing hyde
python -m src.stages.run_stage_7_evaluation --preprocessing decomposition
python -m src.stages.run_stage_7_evaluation --preprocessing graphrag

# Combine with search type
python -m src.stages.run_stage_7_evaluation --search-type keyword --preprocessing hyde
python -m src.stages.run_stage_7_evaluation --search-type hybrid --alpha 0.7 --preprocessing decomposition

# Grid search all combinations (5D: collections × search_types × alphas × strategies × top_k)
python -m src.stages.run_stage_7_evaluation --comprehensive
```

Via UI: Select strategy in Streamlit sidebar dropdown.

## Trade-offs

### None (Baseline)
- **Pros**: Zero latency, no API cost, deterministic
- **Cons**: Query-document vocabulary mismatch hurts recall
- **Use when**: Simple factual queries, debugging

### HyDE
- **Pros**: Bridges semantic gap, good for vague queries
- **Cons**: One LLM call, hypothetical may mismatch corpus style
- **Use when**: Questions with implicit context

### Decomposition
- **Pros**: Handles complex multi-part questions
- **Cons**: Sub-query quality depends on LLM
- **Use when**: Comparison or multi-aspect queries

### GraphRAG
- **Pros**: Cross-document synthesis, entity relationships
- **Cons**: Requires Neo4j, complex setup, higher latency
- **Use when**: "Big picture" questions across documents
