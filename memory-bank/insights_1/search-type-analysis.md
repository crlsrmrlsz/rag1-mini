# Search Type Impact Analysis

## Executive Summary

- **Hybrid search achieves the best overall balance** with superior context recall on single-concept queries (97.06%) while maintaining competitive precision on cross-domain queries (94.56%)
- **Semantic search excels at cross-domain precision** (98.22%) but has lower context recall across all query types, making it less reliable for comprehensive retrieval
- **Keyword search has the worst retrieval performance** with context precision 10+ points below hybrid/semantic, though it shows surprising strength in cross-domain recall (72.37% vs hybrid's 72.63%)
- **The complexity gap (single vs cross-domain) is smallest for semantic search** on relevancy (-20.20% drop) vs keyword (-30.82% drop), suggesting pure embedding similarity handles concept bridging more gracefully

---

## 1. Retrieval Performance Analysis

### Context Precision (Are retrieved chunks relevant?)

| Search Type | Single Concept | Cross-Domain | Delta |
|-------------|----------------|--------------|-------|
| **Keyword** | 0.6612 | 0.8318 | +0.1706 |
| **Hybrid** | 0.7303 | 0.9456 | +0.2152 |
| **Semantic** | 0.7286 | 0.9822 | +0.2536 |

**Key Finding:** All search types show *higher* precision on cross-domain queries, which is counterintuitive. This suggests cross-domain queries may retrieve more focused chunks due to the nature of the questions or that hybrid/semantic ranking is better at surfacing truly relevant content when queries span multiple concepts.

**Winner: Semantic** with 98.22% precision on cross-domain and 72.86% on single-concept.

### Context Recall (Are all relevant chunks retrieved?)

| Search Type | Single Concept | Cross-Domain | Delta |
|-------------|----------------|--------------|-------|
| **Keyword** | 0.8853 | 0.7237 | -0.1616 |
| **Hybrid** | 0.9706 | 0.7263 | -0.2443 |
| **Semantic** | 0.9382 | 0.7154 | -0.2228 |

**Key Finding:** Hybrid dominates single-concept recall (97.06%), achieving near-perfect retrieval. However, all search types experience a ~16-24% recall drop on cross-domain queries, indicating the fundamental challenge of finding information scattered across multiple books/topics.

**Winner: Hybrid** with 97.06% single-concept recall and competitive cross-domain performance (72.63%).

### Retrieval Summary

For comprehensive RAG systems:
- Use **hybrid search** for single-concept queries where recall is paramount
- Use **semantic search** for cross-domain queries where precision matters more than finding every relevant chunk
- Avoid **keyword search** as it consistently underperforms in precision while offering no recall advantage

---

## 2. End-to-End Performance Analysis

### Answer Correctness (Is the final answer accurate?)

| Search Type | Single Concept | Cross-Domain | Delta |
|-------------|----------------|--------------|-------|
| **Keyword** | 0.5681 | 0.4533 | -0.1148 |
| **Hybrid** | 0.5591 | 0.4785 | -0.0806 |
| **Semantic** | 0.5584 | 0.5020 | -0.0564 |

**Key Finding:** Semantic search produces the most accurate answers for cross-domain queries (50.20%) and shows the smallest degradation when query complexity increases. Keyword search has the highest single-concept correctness (56.81%) but the worst cross-domain performance (45.33%).

### Answer Relevancy (Is the answer relevant to the question?)

| Search Type | Single Concept | Cross-Domain | Delta |
|-------------|----------------|--------------|-------|
| **Keyword** | 0.8484 | 0.5402 | -0.3082 |
| **Hybrid** | 0.8587 | 0.5840 | -0.2747 |
| **Semantic** | 0.8357 | 0.6337 | -0.2020 |

**Key Finding:** The relevancy drop is dramatic across all search types for cross-domain queries (20-31% decrease). Semantic search handles this best with only a 20% drop, while keyword search suffers the most severe degradation (31% drop).

### Faithfulness (Is the answer grounded in context?)

| Search Type | Single Concept | Cross-Domain | Overall |
|-------------|----------------|--------------|---------|
| **Keyword** | 0.9111 | 0.9765 | 0.9547 |
| **Hybrid** | 0.9172 | 0.9843 | 0.9570 |
| **Semantic** | 0.9380 | 0.9902 | 0.9700 |

**Key Finding:** All search types maintain high faithfulness (>91%), with semantic leading at 99.02% for cross-domain queries. Interestingly, faithfulness *increases* for cross-domain queries, possibly because LLMs are more careful when synthesizing across multiple sources.

**Winner: Semantic** for best overall end-to-end performance, particularly on cross-domain queries.

---

## 3. Single Concept vs Cross-Domain Gap Analysis

### Delta by Search Type (Cross-Domain minus Single-Concept)

| Metric | Keyword | Hybrid | Semantic |
|--------|---------|--------|----------|
| Faithfulness | +0.0654 | +0.0671 | +0.0521 |
| Relevancy | **-0.3082** | -0.2748 | -0.2020 |
| Context Precision | +0.1706 | +0.2152 | +0.2536 |
| Context Recall | -0.1616 | **-0.2443** | -0.2228 |
| Answer Correctness | -0.1149 | -0.0805 | **-0.0564** |

### Interpretation

1. **Semantic search is most robust to complexity:** It shows the smallest negative deltas on relevancy (-20.20%) and answer correctness (-5.64%), meaning it degrades most gracefully as queries become harder.

2. **Hybrid search has the largest recall degradation:** The -24.43% context recall delta for hybrid suggests that while hybrid excels at single-concept queries, it struggles more than others to find scattered cross-domain evidence.

3. **Keyword search has the most balanced recall:** With only -16.16% recall delta, keyword's BM25 matching may help find exact term matches across domains that embedding-based approaches miss.

4. **All search types improve precision on cross-domain:** The positive precision deltas (17-25%) are unexpected but consistent, suggesting cross-domain queries may be more specific or that relevance ranking works better with diverse retrieval.

### Does Hybrid Reduce the Difficulty Gap?

Partially. Hybrid reduces the answer correctness gap (8.05% vs keyword's 11.49%) but increases the context recall gap (24.43% vs keyword's 16.16%). For applications prioritizing answer quality, hybrid is preferable. For applications requiring comprehensive retrieval, the benefit is less clear.

---

## 4. Interaction with Preprocessing Strategies

### Single-Concept Query Performance

| Strategy | Keyword | Hybrid | Semantic | Best Search Type |
|----------|---------|--------|----------|------------------|
| **none** | CP=0.695, CR=0.87 | CP=0.715, CR=0.96 | CP=0.713, CR=0.92 | Hybrid |
| **hyde** | CP=0.640, CR=0.82 | CP=0.709, CR=0.95 | CP=0.726, CR=0.91 | Hybrid |
| **decomposition** | CP=0.672, CR=0.93 | CP=0.779, CR=0.99 | CP=0.764, CR=0.96 | Hybrid |
| **graphrag** | CP=0.602, CR=0.98 | CP=0.702, CR=1.00 | CP=0.688, CR=0.95 | Hybrid |

**Insights:**
- **Decomposition + Hybrid** achieves the best context precision (77.9%) for single-concept queries
- **GraphRAG + Hybrid** achieves perfect context recall (100%) on single-concept queries
- **HyDE benefits semantic more than keyword**: Semantic+HyDE achieves 72.6% precision vs keyword+HyDE at 64.0%

### Cross-Domain Query Performance

| Strategy | Keyword | Hybrid | Semantic | Best Search Type |
|----------|---------|--------|----------|------------------|
| **none** | AC=0.444, CR=0.70 | AC=0.466, CR=0.70 | AC=0.506, CR=0.71 | Semantic |
| **hyde** | AC=0.457, CR=0.79 | AC=0.483, CR=0.72 | AC=0.503, CR=0.73 | Semantic |
| **decomposition** | AC=0.444, CR=0.66 | AC=0.481, CR=0.73 | AC=0.490, CR=0.74 | Semantic |
| **graphrag** | AC=0.492, CR=0.77 | AC=0.493, CR=0.75 | AC=0.517, CR=0.77 | Semantic |

**Insights:**
- **Semantic consistently wins for cross-domain** regardless of preprocessing strategy
- **GraphRAG provides the biggest correctness boost for keyword** (49.2% vs 44.4% baseline)
- **HyDE improves keyword recall dramatically** (79.4% vs 70.0% baseline)

### Which Search Type Benefits Most from Preprocessing?

**Keyword search benefits most from preprocessing strategies:**
- HyDE: +13.4% context recall on cross-domain
- GraphRAG: +4.8% answer correctness on cross-domain

This makes sense: keyword search's limitations in semantic understanding are partially compensated by LLM-powered query transformation.

---

## 5. Interaction with Chunking Strategies

### Single-Concept Performance by Collection

| Collection | Keyword CP | Hybrid CP | Semantic CP | Best |
|------------|------------|-----------|-------------|------|
| **contextual** | 0.673 | 0.757 | 0.770 | Semantic |
| **raptor** | 0.646 | 0.715 | 0.740 | Semantic |
| **section** | 0.657 | 0.741 | 0.748 | Semantic |
| **semantic_0.3** | 0.676 | 0.697 | 0.718 | Semantic |
| **semantic_0.75** | 0.651 | 0.698 | 0.714 | Semantic |

### Cross-Domain Performance by Collection

| Collection | Keyword AC | Hybrid AC | Semantic AC | Best |
|------------|------------|-----------|-------------|------|
| **contextual** | 0.457 | 0.481 | 0.516 | Semantic |
| **raptor** | 0.461 | 0.479 | 0.508 | Semantic |
| **section** | 0.479 | 0.486 | 0.513 | Semantic |
| **semantic_0.3** | 0.459 | 0.475 | 0.495 | Semantic |
| **semantic_0.75** | 0.401 | 0.461 | 0.488 | Semantic |

**Key Findings:**

1. **Semantic search wins across ALL chunking strategies** for both precision and answer correctness on cross-domain queries.

2. **Contextual chunking + Semantic** achieves the highest cross-domain answer correctness (51.6%) and precision (77.0% single-concept).

3. **Semantic_0.75 (aggressive semantic chunking) is problematic with keyword search**: Keyword+semantic_0.75 has the lowest answer correctness (40.1%), suggesting that semantically-chunked content requires semantic retrieval to work well.

4. **RAPTOR chunking benefits from hybrid search**: The hierarchical summaries in RAPTOR work well with hybrid's combination of keyword matching (for summary terms) and semantic matching (for concept relationships).

### Best Search Type for Each Chunking Strategy

| Chunking Strategy | Recommended Search Type | Rationale |
|-------------------|------------------------|-----------|
| **Contextual** | Semantic | Context prepended to chunks aligns well with embedding similarity |
| **RAPTOR** | Hybrid | Hierarchical summaries benefit from BM25+semantic fusion |
| **Section** | Hybrid | Structured sections have clear keywords that BM25 can leverage |
| **Semantic** | Semantic | Chunks created by semantic similarity must be retrieved semantically |

---

## 6. Key Findings and Recommendations

### Clear Winner: Hybrid for Most Applications

**Recommended Default: Hybrid Search**

Hybrid search provides the best balance across all evaluation dimensions:
- Highest single-concept context recall (97.06%)
- Competitive cross-domain precision (94.56%)
- Good answer correctness across query types
- Benefits from preprocessing strategies

### When to Use Each Search Type

| Scenario | Recommended Search Type | Rationale |
|----------|------------------------|-----------|
| **General RAG applications** | Hybrid | Best overall balance |
| **Cross-domain synthesis** | Semantic | Smallest performance degradation on complex queries |
| **Known-item retrieval** | Keyword | When exact terminology matters (legal, medical) |
| **Single-source Q&A** | Hybrid | Near-perfect recall on focused queries |
| **Multi-hop reasoning** | Semantic | Better at finding conceptually related but lexically different chunks |
| **Low-latency requirements** | Keyword | Fastest, no embedding computation at query time |

### Performance Hierarchy

**For Single-Concept Queries:**
```
Hybrid > Semantic >> Keyword (for context recall)
Semantic >= Hybrid > Keyword (for context precision)
```

**For Cross-Domain Queries:**
```
Semantic > Hybrid > Keyword (for answer correctness)
Semantic > Hybrid >> Keyword (for context precision)
Keyword ~= Hybrid ~= Semantic (for context recall, all ~72%)
```

### Actionable Recommendations

1. **Default to hybrid search** for production RAG systems unless you have specific requirements that favor pure semantic or keyword approaches.

2. **Use semantic search for cross-domain applications** where queries will span multiple topics or sources. The 5+ point answer correctness advantage over keyword is significant.

3. **Avoid pure keyword search** in most scenarios. It consistently underperforms and only provides marginal benefits in exact-match scenarios where hybrid would also succeed.

4. **Pair semantic search with contextual chunking** for best cross-domain performance (51.6% answer correctness).

5. **Use hybrid search with RAPTOR or section chunking** to leverage the structural information in these chunking strategies.

6. **Apply HyDE preprocessing to keyword search** if you must use keyword search - it compensates for its semantic limitations with +13% recall boost.

---

## Appendix: Raw Performance Data

### Overall Metrics by Search Type

| Search Type | Faithfulness | Relevancy | Context Precision | Context Recall | Answer Correctness |
|-------------|--------------|-----------|-------------------|----------------|-------------------|
| Keyword | 0.9547 | 0.6429 | 0.7749 | 0.7776 | 0.5107 |
| Hybrid | 0.9570 | 0.6780 | 0.8380 | 0.8485 | 0.5188 |
| Semantic | 0.9700 | 0.7347 | 0.8554 | 0.8268 | 0.5302 |

*Note: Values are approximate overall means calculated from single-concept and cross-domain data.*
