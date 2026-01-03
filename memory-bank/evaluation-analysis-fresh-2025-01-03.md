# Fresh Evaluation Analysis - January 3, 2025

Analysis of 172 trace files from comprehensive evaluation (102 configurations x 15 questions).

## Question Types

- **Single Concept (5)**: Simple lookup in one book (e.g., "What is serotonin synthesized from?")
- **Cross Domain (10)**: Multi-source synthesis across neuroscience AND philosophy (e.g., "What determines human choices?")

---

## Key Finding 1: Single vs Cross-Domain Performance Gap

| Metric | Single Concept | Cross Domain | Delta |
|--------|---------------|--------------|-------|
| Faithfulness | 0.738 | 0.745 | +0.007 |
| **Relevancy** | **0.659** | **0.408** | **-0.251** |
| Context Precision | 0.470 | 0.618 | +0.148 |
| **Context Recall** | **0.754** | **0.578** | **-0.176** |
| Answer Correctness | 0.462 | 0.396 | -0.067 |

**Interpretation**:
- Cross-domain questions have HIGHER context precision (retrieved chunks are more relevant)
- Cross-domain questions have LOWER context recall (harder to find ALL relevant sources)
- Relevancy drops significantly for cross-domain (answers less aligned with question intent)
- Faithfulness is comparable (LLM stays grounded in both cases)

---

## Key Finding 2: Collection Performance Diverges by Question Type

### Single Concept Performance
| Collection | Faithfulness | Relevancy | Ctx Precision | Ctx Recall |
|------------|-------------|-----------|---------------|------------|
| section | 0.918 | 0.880 | 0.625 | 0.936 |
| semantic_0_3 | 0.902 | 0.830 | 0.734 | 0.933 |
| semantic_0_75 | 0.854 | 0.830 | 0.712 | 0.861 |
| contextual | 0.653 | 0.549 | 0.309* | 0.671 |
| raptor | 0.612 | 0.524 | 0.438 | 0.618 |

### Cross Domain Performance
| Collection | Faithfulness | Relevancy | Ctx Precision | Ctx Recall |
|------------|-------------|-----------|---------------|------------|
| section | 0.991 | 0.586 | 0.927 | 0.763 |
| semantic_0_3 | 0.984 | 0.591 | 0.936 | 0.693 |
| semantic_0_75 | 0.967 | 0.492 | 0.877 | 0.556 |
| raptor | 0.630 | 0.390 | 0.603 | 0.491 |
| contextual | 0.601 | 0.295 | 0.389* | 0.529 |

*contextual has 57% zero context_precision scores - data quality issue

**Insights**:
1. **section** and **semantic** collections excel at both question types
2. **contextual** underperforms dramatically on both (despite being theoretically superior)
3. **raptor** (hierarchical summarization) doesn't help - middle-of-pack
4. Cross-domain drops context_recall across ALL collections (expected - harder to find all sources)

---

## Key Finding 3: Preprocessing Strategies Have Uniform Behavior

### Strategy Delta (Cross - Single)
| Strategy | dFaithfulness | dRelevancy | dCtx Precision | dCtx Recall |
|----------|---------------|------------|----------------|-------------|
| none | -0.005 | **-0.264** | +0.149 | **-0.198** |
| hyde | +0.030 | **-0.173** | +0.186 | **-0.077** |
| decomposition | -0.001 | **-0.288** | +0.120 | **-0.238** |
| graphrag | +0.005 | **-0.274** | +0.135 | **-0.171** |

**Insights**:
1. ALL strategies show same pattern: relevancy drops, ctx_precision rises for cross-domain
2. **HyDE shows smallest degradation** on cross-domain (-0.173 relevancy, -0.077 ctx_recall)
3. **Decomposition hurts cross-domain the most** (-0.288 relevancy, -0.238 ctx_recall)
4. This contradicts intuition: decomposition SHOULD help cross-domain by breaking into sub-queries

---

## Key Finding 4: Alpha (Vector Weight) Matters More for Cross-Domain

### Single Concept by Alpha
| Alpha | Faithfulness | Relevancy | Ctx Precision | Ctx Recall |
|-------|-------------|-----------|---------------|------------|
| 0.0 (keyword) | 0.722 | 0.641 | 0.460 | 0.705 |
| 0.5 (hybrid) | 0.725 | 0.664 | 0.445 | 0.778 |
| 1.0 (vector) | 0.759 | 0.676 | 0.590 | 0.760 |

### Cross Domain by Alpha
| Alpha | Faithfulness | Relevancy | Ctx Precision | Ctx Recall |
|-------|-------------|-----------|---------------|------------|
| 0.0 (keyword) | 0.763 | 0.388 | 0.568 | 0.571 |
| 0.5 (hybrid) | 0.726 | 0.394 | 0.607 | 0.561 |
| 1.0 (vector) | **0.802** | **0.513** | **0.795** | **0.579** |

**Insights**:
1. Pure vector search (alpha=1.0) significantly better for cross-domain
2. Context precision jumps from 0.568 → 0.795 with pure vector
3. Relevancy jumps from 0.388 → 0.513 with pure vector
4. For single-concept, difference is marginal

---

## Key Finding 5: TopK Has Minimal Impact

| TopK | Single Ctx Recall | Cross Ctx Recall |
|------|-------------------|------------------|
| 10 | 0.752 | 0.551 |
| 20 | 0.754 | 0.606 |

TopK=20 helps cross-domain (+0.055 ctx_recall) but difference is small.

---

## Recommendations

### For Single-Concept Questions (Simple Lookups)
1. Use **section** or **semantic** chunking
2. Any preprocessing strategy works (none is fine)
3. Alpha doesn't matter much

### For Cross-Domain Questions (Synthesis)
1. Use **section** or **semantic** chunking
2. Use **HyDE** preprocessing (smallest degradation)
3. Use **pure vector search** (alpha=1.0)
4. Increase top_k to 20

### Avoid
1. **contextual** chunking - consistently underperforms
2. **raptor** - hierarchical summaries don't help retrieval
3. **decomposition** for cross-domain - actually hurts

---

## Data Quality Issues

The **contextual** collection shows 57% zero context_precision scores for single-concept questions. This indicates either:
1. Chunking strategy produces poorly formatted chunks
2. LLM-added context interferes with RAGAS evaluation
3. Ground truth alignment issues

This should be investigated before drawing conclusions about contextual retrieval.

---

## Why Decomposition Fails for Cross-Domain

Counter-intuitively, decomposition (breaking questions into sub-queries) HURTS cross-domain performance. Possible explanations:

1. **Sub-query interference**: Individual sub-queries may retrieve overlapping or redundant chunks
2. **RRF dilution**: Reciprocal Rank Fusion may down-weight important chunks that only appear in one sub-query
3. **Context fragmentation**: Sub-queries may each find partial answers instead of holistic chunks
4. **Synthesis burden**: Answer generation must synthesize across fragmented context

The cross-domain questions already require multi-source synthesis. Adding decomposition creates a double-synthesis problem (decompose → retrieve → synthesize → re-synthesize).
