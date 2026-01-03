# RAGLab Evaluation Findings v3

**Date:** January 3, 2026
**Configurations:** 84 (4 chunking x 3 search x 4 preprocessing x 2 top_k, excluding semantic_0.75)
**Questions:** 15 (5 single-concept + 10 cross-domain)

---

## Experimental Design

**5-Dimensional Grid:**

| Dimension | Values |
|-----------|--------|
| **Chunking** | section, contextual, raptor, semantic_0.3 |
| **Search Type** | keyword (BM25, α=0.0), hybrid (α=0.5), semantic (vector, α=1.0) |
| **Preprocessing** | none, HyDE, decomposition, GraphRAG |
| **Top-K** | 10, 20 |

**Metrics:**
- **Context Recall**: Did we find all relevant information? (retrieval completeness)
- **Context Precision**: Was the retrieved context relevant? (retrieval relevance)
- **Answer Correctness**: Is the final answer correct? (end-to-end quality)

**Statistical Threshold:** Findings require Cohen's d >= 0.8 (large effect size) to account for small sample variance (n=15 questions).

---

## Key Findings

### Finding 1: Different Metrics Have Different Levers

The most important insight from this evaluation: **recall and precision are controlled by different configuration dimensions**.

| To Improve... | Primary Lever | Effect Size |
|---------------|---------------|-------------|
| **Precision** | Search Type (use vector search) | d=2.65 |
| **Recall** | Chunking Strategy (use contextual) | d=1.45 |
| **Recall** | Top-K (use k=20) | d=1.10 |
| **Recall** | Preprocessing (use HyDE over decomposition) | d=1.01 |

**Why this matters:** Practitioners can optimize recall and precision somewhat independently. Want better precision? Use semantic search. Want better recall? Focus on chunking and top_k.

---

### Finding 2: Semantic Search Dramatically Improves Precision

| Search Type | Context Precision | Cohen's d vs Keyword |
|-------------|-------------------|----------------------|
| semantic (α=1.0) | 0.897 ± 0.033 | **+2.65** |
| hybrid (α=0.5) | 0.879 ± 0.027 | **+2.37** |
| keyword (α=0.0) | 0.787 ± 0.048 | baseline |

**Effect:** +11.0% precision gain from keyword to semantic search (d=2.65, highly significant).

**Explanation:** Vector search computes semantic similarity between query and chunks, naturally surfacing the most relevant content. BM25 (keyword) matches terms regardless of semantic context, leading to irrelevant matches. This is the largest effect in the entire study.

---

### Finding 3: Contextual Chunking Provides Best Recall

| Chunking | Context Recall | Cohen's d vs semantic_0.3 |
|----------|----------------|---------------------------|
| contextual | 0.850 ± 0.053 | **+1.45** |
| raptor | 0.830 ± 0.062 | +0.97 |
| section | 0.819 ± 0.056 | +0.83 |
| semantic_0.3 | 0.773 ± 0.054 | baseline |

**Effect:** +7.7% recall gain from semantic_0.3 to contextual chunking (d=1.45, highly significant).

**Explanation:** Contextual chunking prepends LLM-generated context to each chunk, improving retrieval by adding document-level semantics. Semantic chunking with aggressive threshold (0.3) creates small, isolated chunks that lack cross-reference information.

**Note:** contextual, raptor, and section are statistically indistinguishable from each other (d < 0.6). All three outperform semantic_0.3.

---

### Finding 4: The HyDE Cross-Domain Effect

**HyDE uniquely reduces cross-domain penalty:**

| Preprocessing | Single Recall | Cross Recall | Gap | Cross-Domain Penalty |
|---------------|---------------|--------------|-----|----------------------|
| **hyde** | 0.904 | 0.818 | -0.086 | **-9.5%** |
| none | 0.946 | 0.758 | -0.187 | -19.8% |
| graphrag | 0.975 | 0.761 | -0.214 | -21.9% |
| decomposition | 0.975 | 0.694 | -0.281 | **-28.8%** |

**HyDE vs Decomposition for cross-domain recall:** d=1.65 (highly significant)

**Explanation:** HyDE generates a hypothetical answer document before retrieval, creating semantic bridges between domains. For cross-domain questions requiring synthesis of neuroscience AND philosophy, HyDE's hypothetical document contains vocabulary from both domains, improving retrieval from both.

---

### Finding 5: The Decomposition Paradox

Query decomposition shows an inverted pattern:
- **Best for single-concept:** 0.975 recall (tied with graphrag)
- **Worst for cross-domain:** 0.694 recall

| Metric | Decomposition vs None |
|--------|----------------------|
| Single-concept recall | +2.9% (decomposition wins) |
| Cross-domain recall | -8.5% (decomposition loses) |

**Explanation:** Decomposition breaks queries into sub-queries. For single-concept questions, sub-queries are more focused and retrieve better. For cross-domain questions, sub-queries fragment the holistic intent—sub-queries target EITHER neuroscience OR philosophy, but not the synthesis between them.

**Implication:** Use decomposition for focused, single-source queries. Avoid it for synthesis-requiring, multi-source queries.

---

### Finding 6: Top-K=20 Significantly Outperforms Top-K=10 for Recall

| Top-K | Context Recall | Cohen's d |
|-------|----------------|-----------|
| 20 | 0.850 ± 0.045 | **+1.10** |
| 10 | 0.791 ± 0.062 | baseline |

**Effect:** +6.0% recall gain from k=10 to k=20 (d=1.10, highly significant).

**Explanation:** More retrieved chunks means more opportunities to find relevant information. The cost is slightly more noise, but precision differences between k=10 and k=20 are negligible (d=0.04).

**Recommendation:** Default to k=20 unless latency is critical.

---

## Statistically Significant Comparisons (d >= 0.8)

All 8 findings meeting the statistical significance threshold:

| Metric | Comparison | Winner | Gap | d |
|--------|------------|--------|-----|---|
| Context Precision | semantic vs keyword | semantic | +0.110 | **+2.65** |
| Context Precision | hybrid vs keyword | hybrid | +0.092 | **+2.37** |
| Context Recall | contextual vs semantic_0.3 | contextual | +0.077 | **+1.45** |
| Context Recall | k=20 vs k=10 | k=20 | +0.060 | **+1.10** |
| Context Recall | hyde vs decomposition | hyde | +0.059 | **+1.01** |
| Context Recall | raptor vs semantic_0.3 | raptor | +0.057 | **+0.97** |
| Context Recall | graphrag vs decomposition | graphrag | +0.045 | **+0.83** |
| Context Recall | section vs semantic_0.3 | section | +0.046 | **+0.83** |

---

## End-to-End Results (Answer Correctness)

Answer correctness measures final answer quality (dependent on generation LLM: Claude 3.5 Sonnet).

**Top 5 Configurations:**

| Rank | Config | Answer Correctness |
|------|--------|-------------------|
| 1 | section + hyde + semantic + k=20 | **0.606** |
| 2 | contextual + graphrag + hybrid + k=20 | 0.602 |
| 3 | contextual + decomposition + keyword + k=20 | 0.598 |
| 4 | raptor + none + hybrid + k=20 | 0.589 |
| 5 | semantic_0.3 + hyde + semantic + k=20 | 0.584 |

**Observations:**
- No single dimension dominates end-to-end results (all d < 0.8)
- Top configurations span multiple chunking strategies
- k=20 appears in all top-5 (consistent with recall finding)
- GraphRAG shows slightly higher answer correctness than other preprocessing (0.532 vs 0.505-0.515), but d=0.7, just below threshold

---

## Practical Recommendations

### Best All-Around Configuration
```
contextual + hyde + semantic (α=1.0) + k=20
```
- Context Recall: 0.877
- Context Precision: 0.954
- Strong on both single-concept and cross-domain

### Best for Cross-Domain Questions
```
contextual + hyde + semantic (α=1.0) + k=10
```
- Highest cross-domain recall: 0.930
- HyDE's bridging effect is essential

### Simplest Good Configuration (No LLM Preprocessing)
```
section + none + semantic (α=1.0) + k=20
```
- Answer Correctness: 0.563
- Zero preprocessing cost, 93% of best end-to-end performance

### Configuration to Avoid for Cross-Domain
```
* + decomposition + * + *
```
- Decomposition fragments cross-domain intent
- 28.8% penalty on cross-domain recall

---

## Annex: Full Dimension Analysis

### A.1 Context Recall by Dimension (All Questions)

| Dimension | Best | Value | Worst | Value | Gap | d |
|-----------|------|-------|-------|-------|-----|---|
| Chunking | contextual | 0.850 | semantic_0.3 | 0.773 | +0.077 | 1.45 |
| Top-K | 20 | 0.850 | 10 | 0.791 | +0.059 | 1.10 |
| Preprocessing | hyde | 0.847 | decomposition | 0.788 | +0.059 | 1.01 |
| Search | hybrid | 0.828 | keyword | 0.814 | +0.014 | 0.21 |

### A.2 Context Precision by Dimension (All Questions)

| Dimension | Best | Value | Worst | Value | Gap | d |
|-----------|------|-------|-------|-------|-----|---|
| Search | semantic | 0.897 | keyword | 0.787 | +0.110 | 2.65 |
| Preprocessing | decomposition | 0.867 | graphrag | 0.836 | +0.031 | 0.49 |
| Chunking | semantic_0.3 | 0.869 | section | 0.848 | +0.021 | 0.33 |
| Top-K | 10 | 0.856 | 20 | 0.853 | +0.003 | 0.04 |

### A.3 Single-Concept vs Cross-Domain Comparison

**Context Recall:**
| Preprocessing | Single | Cross | Gap |
|---------------|--------|-------|-----|
| hyde | 0.904 | 0.818 | -9.5% |
| none | 0.946 | 0.758 | -19.8% |
| graphrag | 0.975 | 0.761 | -21.9% |
| decomposition | 0.975 | 0.694 | -28.8% |

**Context Precision (inverted pattern):**
| Preprocessing | Single | Cross | Gap |
|---------------|--------|-------|-----|
| hyde | 0.686 | 0.952 | +38.9% |
| none | 0.703 | 0.911 | +29.7% |
| graphrag | 0.664 | 0.922 | +38.9% |
| decomposition | 0.749 | 0.927 | +23.8% |

**Precision Inversion Explanation:** Cross-domain questions match more chunks naturally (content from multiple books is relevant), so precision is higher. Single-concept questions have narrower ground truth, making precision harder.

### A.4 Variance Analysis (Coefficient of Variation)

Lower CV = more consistent results across other dimensions.

| Dimension | Lowest CV (Most Consistent) | Highest CV (Most Variable) |
|-----------|----------------------------|---------------------------|
| Chunking (recall) | contextual (0.062) | raptor (0.075) |
| Search (precision) | hybrid (0.030) | keyword (0.061) |
| Preprocessing (recall) | graphrag (0.051) | none (0.078) |

**GraphRAG shows most consistent recall** across chunking/search combinations.

---

## Limitations

1. **Sample size:** 15 questions limits statistical power; effects < d=0.8 may not replicate
2. **Single corpus:** Neuroscience + philosophy books may not generalize
3. **Single LLM:** End-to-end results specific to Claude 3.5 Sonnet
4. **No cost tracking:** LLM API costs not measured

---

*v3 Changes from v2:*
- *Renamed search types: keyword (α=0.0), hybrid (α=0.5), semantic (α=1.0)*
- *Separated single-concept vs cross-domain analysis*
- *Added explanations for WHY each finding occurs*
- *Stricter statistical threshold (only d >= 0.8)*
- *Reduced to 6 well-grounded findings from 10+ loosely-grounded*
- *Added variance analysis in annex*

*Generated January 3, 2026*
