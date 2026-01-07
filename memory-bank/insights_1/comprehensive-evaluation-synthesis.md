# RAG Pipeline Comprehensive Evaluation Synthesis

**Date:** 2026-01-03
**Source:** `statistics_report_20260101_164236.json` (102 combinations, 15 questions)

---

## Overview

This synthesis integrates findings from three specialized analyses:
- [Chunking Strategy Analysis](chunking-strategy-analysis.md) - 5 chunking approaches
- [Preprocessing Strategy Analysis](preprocessing-strategy-analysis.md) - 4 query preprocessing methods
- [Search Type Analysis](search-type-analysis.md) - 3 retrieval methods

### Evaluation Dimensions
| Dimension | Options | Primary Impact |
|-----------|---------|----------------|
| Chunking | Contextual, RAPTOR, Section, Semantic 0.3, Semantic 0.75 | Context Recall, Answer Correctness |
| Preprocessing | none, hyde, decomposition, graphrag | Answer Correctness |
| Search Type | keyword, hybrid, semantic | Context Precision |
| Top K | 10, 20 | Marginal |

---

## Key Metrics Summary

### Single Concept Queries (5 questions)

| Best Configuration | Metric | Score |
|-------------------|--------|-------|
| Contextual + graphrag + hybrid | **Answer Correctness** | 0.687 |
| Semantic 0.3 + decomposition + semantic | Context Precision | 0.963 |
| Contextual + graphrag + hybrid | Context Recall | 1.000 |
| Semantic 0.75 + hyde + keyword | Relevancy | 0.931 |

### Cross-Domain Queries (10 questions)

| Best Configuration | Metric | Score |
|-------------------|--------|-------|
| RAPTOR + none + semantic | **Answer Correctness** | 0.595 |
| RAPTOR + none + semantic | Context Precision | 1.000 |
| Contextual + hyde + semantic | Context Recall | 0.895 |
| Section + hyde + semantic | Relevancy | 0.782 |

---

## Variance Contribution Analysis

Which dimension has the most impact on each metric?

### Single Concept Queries

| Metric | Most Influential | 2nd | 3rd |
|--------|-----------------|-----|-----|
| Context Precision | **search_type** (17%) | strategy (11%) | collection (5%) |
| Context Recall | **search_type** (24%) | collection (22%) | strategy (18%) |
| Answer Correctness | **collection** (23%) | top_k (12%) | strategy (7%) |
| Relevancy | **top_k** (13%) | collection (11%) | strategy (5%) |

### Cross-Domain Queries

| Metric | Most Influential | 2nd | 3rd |
|--------|-----------------|-----|-----|
| Context Precision | **search_type** (92%) | collection (9%) | strategy (4%) |
| Context Recall | **collection** (65%) | strategy (24%) | top_k (22%) |
| Answer Correctness | **search_type** (22%) | strategy (7%) | collection (6%) |
| Relevancy | **collection** (28%) | search_type (21%) | strategy (4%) |

### Key Insight

**Search type dominates retrieval precision**, especially for cross-domain queries (92% variance contribution). **Collection (chunking) dominates recall and overall answer quality**. Preprocessing strategy has moderate influence across all metrics but is never the dominant factor.

---

## The Optimal Configuration

Based on the integrated analysis, the recommended configuration differs by query type:

### For Single-Concept Queries
```
Chunking:       Contextual
Preprocessing:  graphrag
Search Type:    hybrid
Top K:          20
```
**Expected Performance:**
- Answer Correctness: ~62% (best overall)
- Context Recall: ~100% (perfect)
- Context Precision: ~72%

### For Cross-Domain Queries
```
Chunking:       Contextual (or RAPTOR)
Preprocessing:  graphrag (or none for simplicity)
Search Type:    semantic
Top K:          10-20
```
**Expected Performance:**
- Answer Correctness: ~52% (best available)
- Context Precision: ~98%
- Context Recall: ~79%

### Universal Recommendation
```
Chunking:       Contextual
Preprocessing:  graphrag
Search Type:    hybrid (or semantic if cross-domain heavy)
Top K:          20
```

---

## Critical Insights by Dimension

### 1. Chunking Strategy

| Insight | Evidence |
|---------|----------|
| **Contextual is the clear winner** | Best answer correctness (59.1% single, 48.8% cross) |
| **Recall matters more than precision** | Contextual's high recall (96.3%) correlates with answer quality |
| **Avoid Semantic 0.75** | Underperforms on all metrics |
| **RAPTOR excels at faithfulness** | 95.2% - best for low-hallucination needs |

### 2. Preprocessing Strategy

| Insight | Evidence |
|---------|----------|
| **GraphRAG adds +5.7% answer correctness** | Best single & cross-domain performance |
| **HyDE stabilizes cross-domain retrieval** | Smallest recall drop (-10.5% vs -30.4% for decomposition) |
| **Decomposition fails on cross-domain** | -30.4% context recall degradation |
| **"none" is surprisingly competitive** | For simple queries, preprocessing adds cost without benefit |

### 3. Search Type

| Insight | Evidence |
|---------|----------|
| **Hybrid balances recall and precision** | 97% single-concept recall, 94.6% cross precision |
| **Semantic wins cross-domain** | 50.2% answer correctness (vs 45.3% keyword) |
| **Keyword search is obsolete** | Consistently 10+ points below hybrid/semantic |
| **Search type dominates precision variance** | 92% variance on cross-domain precision |

---

## Single Concept vs Cross-Domain: The Gap

### Performance Degradation (Delta)

| Metric | Average Drop | Best Strategy for Minimizing |
|--------|-------------|------------------------------|
| Context Recall | -17% to -30% | Section chunking (-16.6%) |
| Relevancy | -20% to -31% | Semantic search (-20.2%) |
| Answer Correctness | -5% to -12% | Semantic search (-5.6%) |

### Why Cross-Domain is Hard

1. **Information scattering**: Relevant chunks are distributed across multiple documents/topics
2. **Concept bridging**: Embeddings struggle with cross-domain semantic relationships
3. **Synthesis challenge**: LLMs must integrate diverse contexts coherently

### Best Approaches for Cross-Domain

1. **Use semantic search** - handles concept bridging most gracefully
2. **Use contextual or RAPTOR chunking** - preserve document-level context
3. **Consider HyDE** - hypothetical documents bridge domain gaps
4. **GraphRAG for answer quality** - knowledge graph provides structural relationships

---

## Anti-Patterns to Avoid

| Configuration | Problem |
|---------------|---------|
| Decomposition + Cross-domain queries | 30% context recall drop |
| Semantic 0.75 + any | Underperforms everywhere |
| Keyword + Semantic chunking | Semantically-chunked content needs semantic retrieval |
| Pure keyword search | 10+ points below alternatives on precision |

---

## Synergistic Combinations

| Combination | Why It Works |
|-------------|--------------|
| **Contextual + GraphRAG** | LLM context in chunks + knowledge graph structure |
| **RAPTOR + Hybrid search** | Hierarchical summaries + BM25/semantic fusion |
| **HyDE + Keyword** | Compensates for keyword's semantic limitations (+13% recall) |
| **Semantic search + Contextual chunks** | Best cross-domain answer correctness (51.6%) |

---

## Decision Framework

```
                           Query Type
                    /                    \
            Single-Concept            Cross-Domain
                  |                        |
         Use HYBRID search          Use SEMANTIC search
                  |                        |
       Use GRAPHRAG preprocessing   Use GRAPHRAG or HYDE
                  |                        |
     Use CONTEXTUAL chunking        Use CONTEXTUAL/RAPTOR
                  |                        |
         Expected: ~62% AC            Expected: ~52% AC
```

---

## Recommendations Summary

### Infrastructure Choices (One-Time)

1. **Chunking**: Contextual (requires LLM at index time, but best ROI)
2. **Vector DB**: Weaviate with hybrid search enabled
3. **Knowledge Graph**: Neo4j + GDS for GraphRAG (if answer quality is paramount)

### Query-Time Choices

1. **Simple factual queries**: No preprocessing, hybrid search
2. **Cross-domain synthesis**: GraphRAG or HyDE, semantic search
3. **Multi-step procedural**: Decomposition, hybrid search

### Cost-Performance Trade-offs

| Configuration | Cost Level | Answer Correctness |
|---------------|-----------|-------------------|
| Contextual + none + hybrid | Low | ~56% |
| Contextual + graphrag + hybrid | High (Neo4j) | ~62% |
| Section + none + hybrid | Lowest | ~52% |

---

## Next Steps for Improvement

1. **Query routing**: Implement classifier to route queries to optimal configuration
2. **Reranking**: Add cross-encoder reranking for final chunk selection
3. **Larger evaluation set**: 15 questions provide insights but more data would increase confidence
4. **Alpha tuning**: Test hybrid search alpha values beyond 0.0/0.5/1.0

---

*Generated: 2026-01-03 | Source: RAGLab Comprehensive Evaluation*
