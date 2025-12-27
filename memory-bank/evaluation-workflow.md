# Evaluation Workflow Analysis

This document provides a comprehensive analysis of the RAG1-Mini evaluation system, including strategy taxonomy, retrieval diagrams, and the black-box evaluation philosophy.

## Strategy Taxonomy

The RAG pipeline uses two independent axes of strategies:

### 1. Chunking Strategies (Stage 4 - Index Time)

Chunking strategies determine how documents are split and stored in Weaviate. Each creates a separate collection.

| Strategy | Collection Pattern | Description | Research |
|----------|-------------------|-------------|----------|
| **section** | `RAG_section_*` | Sequential chunking with 2-sentence overlap | Baseline |
| **contextual** | `RAG_contextual_*` | LLM-generated context prepended to chunks | [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval) |
| **raptor** | `RAG_raptor_*` | Hierarchical tree with GMM clustering + summaries | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) |

### 2. Preprocessing Strategies (Evaluation Time - Query Time)

Preprocessing strategies transform queries before retrieval. They work with any chunking collection.

| Strategy | Transform | Retrieval | Research |
|----------|-----------|-----------|----------|
| **none** | Query unchanged | Single hybrid search | Baseline |
| **hyde** | Hypothetical answer | Single search with HyDE passage | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| **decomposition** | 3-4 sub-questions | Multi-query + RRF merge | [arXiv:2507.00355](https://arxiv.org/abs/2507.00355) |
| **graphrag** | Entity extraction | Vector + Neo4j graph hybrid | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) |

## Comprehensive Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPREHENSIVE EVALUATION FLOW                            │
│                    (run_stage_7_evaluation.py --comprehensive)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FOR EACH COMBINATION:                                                      │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐                 │
│  │ Collections  │  │    Alphas      │  │   Strategies    │                 │
│  │ (dynamic)    │  │ [0.0-1.0]     │  │ [none,hyde,     │                 │
│  │              │  │                │  │  decomp,graph]  │                 │
│  └──────────────┘  └────────────────┘  └─────────────────┘                 │
│        │                   │                    │                           │
│        └───────────────────┴────────────────────┘                           │
│                            │                                                │
│                            ▼                                                │
│                   Total combinations = N_collections * 5 * 4                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Strategy-Specific Retrieval Paths

### None Strategy (Baseline)

```
Question ──► No preprocessing ──► Hybrid Search ──► Contexts
                                       │
                                       ▼
                             Weaviate (BM25 + Vector)
                                       │
                                       ▼
                                Optional Reranking
                                       │
                                       ▼
                                  Top-K Results
```

### HyDE Strategy (Hypothetical Document Embeddings)

```
Question ──► LLM generates hypothetical answer ──► Hybrid Search ──► Contexts
                        │                               │
                        ▼                               ▼
           "Procrastination stems from..."    Weaviate (BM25 + Vector)
                                                       │
                                                       ▼
                                              Optional Reranking
                                                       │
                                                       ▼
                                                  Top-K Results
```

**Theory**: HyDE bridges the semantic gap between questions and documents by searching for passages similar to a plausible answer rather than the question itself.

### Decomposition Strategy (Multi-Query + RRF)

```
Question ──► decompose_query() ──► [sub_q1, sub_q2, sub_q3, sub_q4]
                                            │
                                            ▼
                                   ┌───────────────────┐
                                   │ Search sub_q1 ────┼──► Results1
                                   │ Search sub_q2 ────┼──► Results2
                                   │ Search sub_q3 ────┼──► Results3
                                   │ Search sub_q4 ────┼──► Results4
                                   └───────────────────┘
                                            │
                                            ▼
                                      RRF Merge
                                   (k=60 formula)
                                            │
                                            ▼
                                    Optional Reranking
                                            │
                                            ▼
                                      Top-K Contexts
```

**RRF Formula**:
```
RRF_score(d) = sum(1 / (k + rank(d, q))) for each query q
```
where k=60 (standard from literature).

**Theory**: Complex questions are decomposed into simpler sub-questions. Results from each sub-query are merged, with documents appearing in multiple result lists receiving boosted scores.

### GraphRAG Strategy (Hybrid Graph + Vector)

```
Question ──► extract_query_entities() ──► [entity1, entity2, ...]
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
       Vector Search      Neo4j Graph
       (Weaviate)         Traversal
            │                   │
            │                   ▼
            │          Entity → Neighbors → chunk_ids
            │                   │
            └─────────┬─────────┘
                      ▼
               Graph Boost Merge
          (graph-found chunks ranked higher)
                      │
                      ▼
              Optional Reranking
                      │
                      ▼
                Top-K Contexts
```

**Theory**: GraphRAG combines dense vector retrieval with knowledge graph traversal. Chunks that are both semantically similar AND connected via entity relationships receive boosted ranking.

## Black-Box Evaluation Philosophy

Each strategy is evaluated as a **black box**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      BLACK BOX STRATEGY                         │
│                                                                 │
│   Questions ───────────►  [Strategy]  ───────────► Contexts    │
│      (input)                                        (output)    │
│                                                                 │
│   The evaluation ONLY measures:                                 │
│   - Quality of retrieved contexts (context_precision)          │
│   - Faithfulness of generated answers                          │
│   - Relevancy of answers to questions                          │
│                                                                 │
│   The evaluation does NOT care about:                          │
│   - How the strategy works internally                          │
│   - Number of LLM calls                                        │
│   - Latency or cost                                            │
└─────────────────────────────────────────────────────────────────┘
```

This allows fair comparison between fundamentally different strategies (e.g., single-query hyde vs multi-query decomposition vs graph-based graphrag).

## Strategy Comparison Table

| Aspect | none | hyde | decomposition | graphrag |
|--------|------|------|---------------|----------|
| **LLM Calls** | 0 | 1 | 1 | 1 |
| **Search Queries** | 1 | 1 | 3-4 | 1 + graph |
| **Merge Strategy** | N/A | N/A | RRF | Graph boost |
| **External DB** | Weaviate | Weaviate | Weaviate | Weaviate + Neo4j |
| **Best For** | Simple queries | Semantic matching | Multi-aspect questions | Entity-centric queries |

## RAPTOR as Chunking Strategy

RAPTOR is a **chunking strategy**, not a preprocessing strategy. It creates hierarchical summaries at index time:

```
Original Chunks (Level 0)
        │
        ▼
   GMM Clustering
        │
        ▼
   LLM Summarization ──► Level 1 Summaries
        │
        ▼
   GMM Clustering
        │
        ▼
   LLM Summarization ──► Level 2 Summaries
        │
        ... (up to 4 levels)
```

All nodes (leaves + summaries) are stored flat in Weaviate, enabling "collapsed tree" retrieval where both detailed and thematic content can be retrieved.

RAPTOR collections are tested via the collection axis (e.g., `RAG_raptor_embed3large_v1`), not the preprocessing strategy axis.

## Implementation Details

### File: `src/evaluation/ragas_evaluator.py`

The `retrieve_contexts()` function implements strategy-aware retrieval:

```python
def retrieve_contexts(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    use_reranking: bool = True,
    alpha: float = 0.5,
    preprocessed: Optional[PreprocessedQuery] = None,  # Strategy-aware
) -> List[str]:
```

Routing logic:
1. If `preprocessed.strategy_used == "decomposition"`: Execute multi-query RRF
2. If `preprocessed.strategy_used == "graphrag"`: Execute Neo4j hybrid retrieval
3. Otherwise: Standard hybrid search

### File: `src/stages/run_stage_7_evaluation.py`

Comprehensive mode iterates through all combinations:

```python
for collection in collections:        # Chunking strategies
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:  # Hybrid balance
        for strategy in strategies:    # Preprocessing strategies
            run_evaluation(...)
```

## Historical Context

The comprehensive evaluation mode was developed in Phase 5 (Alpha Tuning) before RAPTOR and GraphRAG were implemented. The original implementation only tested `none` and `hyde` strategies, both of which use single-query retrieval.

The strategy-aware retrieval was added to properly support:
- **decomposition**: Multi-query with RRF merge (Phase 4)
- **graphrag**: Neo4j hybrid retrieval (Phase 8)

## Running Comprehensive Evaluation

```bash
# Full grid search
python -m src.stages.run_stage_7_evaluation --comprehensive

# Single strategy test
python -m src.stages.run_stage_7_evaluation --preprocessing decomposition --questions 5

# With specific collection
python -m src.stages.run_stage_7_evaluation \
    --collection RAG_raptor_embed3large_v1 \
    --preprocessing graphrag
```

## Metrics

The evaluation uses RAGAS metrics:

| Metric | Description | Requires Reference |
|--------|-------------|-------------------|
| faithfulness | Is the answer grounded in context? | No |
| relevancy | Does the answer address the question? | No |
| context_precision | Are retrieved chunks relevant? | No |
| context_recall | Did retrieval capture needed info? | Yes |
| factual_correctness | Is the answer correct? | Yes |
| answer_correctness | Weighted F1 + semantic similarity | Yes |
| squad_f1 | Token-level F1 (benchmark comparison) | Yes |

## References

- [RAPTOR Paper](https://arxiv.org/abs/2401.18059) - Hierarchical summarization
- [HyDE Paper](https://arxiv.org/abs/2212.10496) - Hypothetical Document Embeddings
- [Query Decomposition](https://arxiv.org/abs/2507.00355) - Multi-hop retrieval
- [GraphRAG Paper](https://arxiv.org/abs/2404.16130) - Knowledge graph + vector hybrid
- [RAGAS Framework](https://docs.ragas.io/) - LLM-as-judge evaluation
